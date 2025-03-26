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
        for i in range(10):
            self.parsers.append(
                Parser(
                    model_dir=ray_env_model_dir,
                    output_filename=os.path.join(
                        parser_config["export_path"],
                        OperatorName.PARSER.value,
                        f"{self.timestamp}.jsonl",
                    ),
                ).remote()
            )

        self.splitters = []
        splitter_config = self.cfg.process_config[OperatorName.SPLITTER]
        for i in range(10):
            self.splitters.append(
                Splitter(
                    type=splitter_config["type"],
                    chunk_overlap=splitter_config["chunk_overlap"],
                    chunk_size=splitter_config["chunk_size"],
                    model_dir=ray_env_model_dir,
                    output_filename=os.path.join(
                        splitter_config["export_path"],
                        OperatorName.SPLITTER.value,
                        f"{self.timestamp}.jsonl",
                    ),
                ).remote()
            )
        self.embedders = []
        embedder_config = self.cfg.process_config[OperatorName.EMBEDDER]
        for i in range(10):
            self.embedders.append(
                Embedder(
                    model_dir=ray_env_model_dir,
                    output_filename=os.path.join(
                        embedder_config["export_path"],
                        OperatorName.EMBEDDER.value,
                        f"{self.timestamp}.jsonl",
                    ),
                ).remote()
            )

        self.writers = []
        writer_config = self.cfg.process_config[OperatorName.WRITER]
        for i in range(10):
            self.writers.append(
                Writer(
                    rag_endpoint=writer_config["rag_endpoint"],
                    rag_key=writer_config["rag_key"],
                    embed_dims=writer_config["embed_dims"],
                    model_dir=ray_env_model_dir,
                    output_filename=os.path.join(
                        writer_config["export_path"],
                        OperatorName.WRITER.value,
                        f"{self.timestamp}.jsonl",
                    ),
                ).remote()
            )

    def run(self):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        all_tstart = time.time()
        logger.info(f"Loading dataset from {self.cfg.dataset_path} ...")
        input_files = get_input_files(self.cfg.dataset_path, self.cfg.filter_pattern)

        process_results = []
        for i, file in enumerate(input_files):
            logger.info(f"Processing {file}, progress {i+1}/{len(input_files)} ...")
            docs = self.parsers[i % len(self.parsers)].process.remote([file])
            chunks = self.splitters[i % len(self.splitters)].process.remote(docs)
            embedded_chunks = self.embedders[i % len(self.embedders)].process.remote(
                chunks
            )
            result = self.writers[i % len(self.embedders)].process.remote(
                embedded_chunks
            )
            process_results.append(result)

            logger.info(f"Enqueued {file} progress {i+1}/{len(input_files)} ...")

        ray.get(process_results)
        all_tend = time.time()
        logger.info(f"All ops are done in {all_tend - all_tstart:.3f}s.")
