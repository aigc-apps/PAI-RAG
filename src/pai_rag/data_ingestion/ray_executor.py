import os
from pathlib import Path
from typing import List

from pai_rag.data_ingestion.datasource.filedelta_datasource import FileDeltaDatasource
from pai_rag.data_ingestion.models.config.datasource import DataSourceConfig
from pai_rag.data_ingestion.models.config.operator import (
    BaseOperatorConfig,
    EmbedderConfig,
    ParserConfig,
    SplitterConfig,
    WriterConfig,
)
from pai_rag.data_ingestion.operators.base import BaseOperator
from pai_rag.data_ingestion.operators.embedder import Embedder
from pai_rag.data_ingestion.operators.parser import Parser
from pai_rag.data_ingestion.operators.split import Splitter
from pai_rag.data_ingestion.operators.writer import Writer
from pai_rag.data_ingestion.utils.concurrency_utils import compute_concurrency_count
from pai_rag.data_ingestion.utils.dataset_utils import get_input_files

import ray
import time
from loguru import logger

from pai_rag.data_ingestion.utils.filename_utils import BlockFileNameProvider


DEFAULT_WORKING_DIR = "/app"
DEFAULT_ROWS_PER_FILE = 10000


class RayExecutor:
    """
    Executor based on Ray.
    """

    def __init__(self, working_dir: str = DEFAULT_WORKING_DIR):
        self.working_dir = working_dir
        # init ray
        print("model dir: ", os.environ.get("PAI_RAG_MODEL_DIR"))
        if os.environ.get("PAI_RAG_MODEL_DIR"):
            ray_env_model_dir = os.environ["PAI_RAG_MODEL_DIR"]
        else:
            ray_env_model_dir = os.path.join(self.working_dir, "model_repository")
            os.environ["PAI_RAG_MODEL_DIR"] = ray_env_model_dir
        logger.info(
            f"Initing Ray with working_dir: {self.working_dir}, set env: PAI_RAG_MODEL_DIR = {ray_env_model_dir}..."
        )
        ray.init(
            runtime_env={
                "working_dir": self.working_dir,
            }
        )
        self.execution_ts = time.strftime("%Y%m%d-%H%M%S")

    def _resolve_op_class(self, op_config: BaseOperatorConfig) -> BaseOperator:
        if isinstance(op_config, ParserConfig):
            return Parser
        elif isinstance(op_config, SplitterConfig):
            return Splitter
        elif isinstance(op_config, EmbedderConfig):
            return Embedder
        elif isinstance(op_config, WriterConfig):
            return Writer

        raise ValueError(f"Unknown operator config: {op_config}.")

    def _need_batch_execution(self, op_config: BaseOperatorConfig) -> bool:
        return isinstance(op_config, EmbedderConfig) or isinstance(
            op_config, WriterConfig
        )

    def run(
        self,
        op_configs: List[BaseOperatorConfig] = [],
        datasource_config: DataSourceConfig = None,
    ):
        """
        Running the dataset process pipeline.
        """
        start_time = time.time()

        if datasource_config is not None:
            filename_provider = BlockFileNameProvider(
                run_label=f"read-{self.execution_ts}", file_format="jsonl"
            )

            datasource = FileDeltaDatasource(config=datasource_config)
            dataset = ray.data.read_datasource(datasource).materialize()
            Path(datasource_config.output_path).mkdir(parents=True, exist_ok=True)
            dataset.write_json(
                datasource_config.output_path,
                min_rows_per_file=DEFAULT_ROWS_PER_FILE,
                try_create_dir=True,
                filename_provider=filename_provider,
                force_ascii=False,
            )
        else:
            if len(op_configs) == 0:
                logger.warning(
                    "No op_configs and datasource provided, skipping dataset process pipeline."
                )
                return

            input_files_list = get_input_files(
                file_path_or_directory=op_configs[0].input_path,
                filter_pattern="*.jsonl",
            )
            dataset = ray.data.read_json(input_files_list)

        for op_config in op_configs:
            logger.info(f"Executing {op_config.name}...")
            OP_TYPE = self._resolve_op_class(op_config=op_config)
            op_concurrency = op_config.concurrency or compute_concurrency_count(
                num_cpus=op_config.num_cpus,
                memory=op_config.memory,
                num_gpus=op_config.num_gpus,
            )
            if self._need_batch_execution(op_config=op_config):
                # Embedder需要batch执行
                logger.info(
                    f"Executing {op_config.name} in batch mode. Task concurrency: {op_concurrency}"
                )
                dataset = dataset.map_batches(
                    OP_TYPE,
                    batch_size=op_config.batch_size,
                    num_cpus=op_config.num_cpus,
                    num_gpus=op_config.num_gpus,
                    memory=op_config.memory,
                    concurrency=op_concurrency,
                    fn_constructor_kwargs={"config": op_config},
                ).materialize()
            else:
                logger.info(
                    f"Executing {op_config.name} in flat_map mode. Task concurrency: {op_concurrency}"
                )
                dataset = dataset.flat_map(
                    OP_TYPE,
                    num_cpus=op_config.num_cpus,
                    num_gpus=op_config.num_gpus,
                    memory=op_config.memory,
                    concurrency=op_concurrency,
                    fn_constructor_kwargs={"config": op_config},
                ).materialize()

            # 保存op结果，保存向量库无需执行
            if not isinstance(op_config, WriterConfig):
                filename_provider = BlockFileNameProvider(
                    run_label=f"{op_config.name.value}-{self.execution_ts}",
                    file_format="jsonl",
                )
                Path(op_config.output_path).mkdir(parents=True, exist_ok=True)

                dataset.write_json(
                    op_config.output_path,
                    min_rows_per_file=DEFAULT_ROWS_PER_FILE,
                    try_create_dir=True,
                    filename_provider=filename_provider,
                    force_ascii=False,
                )
            logger.info(f"Finished executing {op_config.name}...")

        logger.info(f"All ops are done in {time.time() - start_time:.3f}s.")


ray_executor = RayExecutor()
