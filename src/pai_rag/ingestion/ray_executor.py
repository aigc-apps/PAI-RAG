import os
from pai_rag.ingestion.operators.base import OperatorName
from pai_rag.ingestion.operators.embedder import Embedder
from pai_rag.ingestion.operators.parser import Parser
from pai_rag.ingestion.operators.split import Splitter
from pai_rag.ingestion.operators.writer import Writer
from pai_rag.ingestion.utils.dataset_utils import get_input_files
import ray
import time
from loguru import logger


class RayExecutor:
    """
    Executor based on Ray.

    Run Data-Juicer data processing in a distributed cluster.

        1. Support Filter, Mapper and Exact Deduplicator operators for now.
        2. Only support loading `.json` files.
        3. Advanced functions such as checkpoint, tracer are not supported.

    """

    def __init__(self, cfg=None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        self.cfg = cfg
        # init ray
        if os.environ.get("PAI_RAG_MODEL_DIR"):
            ray_env_model_dir = os.environ["PAI_RAG_MODEL_DIR"]
        else:
            ray_env_model_dir = os.path.join(self.cfg.working_dir, "model_repository")
            os.environ["PAI_RAG_MODEL_DIR"] = ray_env_model_dir
        logger.info(
            f"Initing Ray with working_dir: {self.cfg.working_dir}, set env: PAI_RAG_MODEL_DIR = {ray_env_model_dir}..."
        )
        ray.init(
                runtime_env={
                "working_dir": self.cfg.working_dir,
            }
        )
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.parsers = []
        parser_config = self.cfg.process_config[OperatorName.PARSER]
        concurrency = parser_config.get("concurrency", 10)
        logger.info(f"Creating parser pool with {concurrency} workers...")
        for i in range(concurrency):
            self.parsers.append(
                Parser.options(num_cpus=1).remote(
                    model_dir=ray_env_model_dir,
                    output_filename=os.path.join(
                    parser_config["export_path"],
                    OperatorName.PARSER.value,
                    f"{self.timestamp}_worker{i+1}.jsonl",
                ))
            )

        self.splitters = []
        splitter_config = self.cfg.process_config[OperatorName.SPLITTER]
        concurrency = splitter_config.get("concurrency", 10)
        logger.info(f"Creating splitter pool with {concurrency} workers...")

        for i in range(concurrency):
            self.splitters.append(
                Splitter.options(num_cpus=1).remote(
                    type=splitter_config["type"],
                    chunk_overlap=splitter_config["chunk_overlap"],
                    chunk_size=splitter_config["chunk_size"],
                    model_dir=ray_env_model_dir,
                    output_filename=os.path.join(
                        splitter_config["export_path"],
                        OperatorName.SPLITTER.value,
                        f"{self.timestamp}_worker{i+1}.jsonl",
                    ),
                )
            )
        self.embedders = []
        embedder_config = self.cfg.process_config[OperatorName.EMBEDDER]
        concurrency = embedder_config.get("concurrency", 5)
        num_cpus = embedder_config.get("num_cpus", 1)
        num_gpus = embedder_config.get("num_gpus", 0.33)

        logger.info(f"Creating embedder pool with {concurrency} workers...")
        for i in range(concurrency):
            self.embedders.append(
                Embedder.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(
                    model_dir=ray_env_model_dir,
                    output_filename=os.path.join(
                        embedder_config["export_path"],
                        OperatorName.EMBEDDER.value,
                        f"{self.timestamp}_worker{i+1}.jsonl",
                    ),
                )
            )

        self.writers = []
        writer_config = self.cfg.process_config[OperatorName.WRITER]
        concurrency = writer_config.get("concurrency", 5)
        logger.info(f"Creating writer pool with {concurrency} workers...")
        for i in range(concurrency):
            self.writers.append(
                Writer.options(num_cpus=1).remote(
                    rag_endpoint=writer_config["rag_endpoint"],
                    rag_key=writer_config["rag_key"],
                    embed_dims=writer_config["embed_dims"],
                    knowledgebase=writer_config["knowledgebase"],
                    model_dir=ray_env_model_dir,
                    output_filename=os.path.join(
                        writer_config["export_path"],
                        OperatorName.WRITER.value,
                        f"{self.timestamp}_worker{i+1}.jsonl",
                    ),
                )
            )

    def run(self):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        all_tstart = time.time()
        logger.info(f"Loading dataset from {self.cfg.dataset_path} ...")
        input_files = get_input_files(self.cfg.dataset_path)

        pending_doc_ref_list = []
        process_results = []
        batch_size = self.cfg.batch_size
        for i, file in enumerate(input_files):
            pending_doc_ref_list.append(self.parsers[i % len(self.parsers)].process.remote([file]))
        
        batch_index = 0
        while len(pending_doc_ref_list) > 0:
            num_returns = min(len(pending_doc_ref_list), batch_size)
            done_doc_ref_list, pending_doc_ref_list = ray.wait(pending_doc_ref_list, num_returns=num_returns, timeout=120)
            if len(done_doc_ref_list) == 0:
                logger.warning("No parse task finished")
                continue

            logger.info(f"Get {len(done_doc_ref_list)} task completed.")
            results = ray.get(done_doc_ref_list)
            flattern_docs = [item for sublist in results for item in sublist]
            logger.info(f"Get {len(flattern_docs)} docs parsed.")

            doc_batch = []
            for i, doc in enumerate(flattern_docs):
                doc_batch.append(doc)
                if (i + 1) % batch_size == 0 or i == len(flattern_docs) - 1:
                    chunks = self.splitters[batch_index % len(self.splitters)].process.remote(doc_batch)
                    embedded_chunks = self.embedders[batch_index % len(self.embedders)].process.remote(
                        chunks
                    )
                    result = self.writers[batch_index % len(self.writers)].process.remote(
                        embedded_chunks
                    )
                    process_results.append(result)

                    logger.info(f"Enqueued {batch_index} batch. Progress {i+1}/{len(flattern_docs)} ...")
                    doc_batch = []
                    batch_index += 1

        ray.get(process_results)
        all_tend = time.time()
        logger.info(f"All ops are done in {all_tend - all_tstart:.3f}s.")
