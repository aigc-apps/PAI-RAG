import os
from typing import List

import psutil
from pai_rag.data_ingestion.datasource.filedelta_datasource import FileDeltaDatasource
from pai_rag.data_ingestion.models.config.datasource import DataSourceConfig
from pai_rag.data_ingestion.models.config.operator import BaseOperatorConfig, EmbedderConfig, ParserConfig, SplitterConfig
from pai_rag.data_ingestion.operators.base import BaseOperator
from pai_rag.data_ingestion.operators.embedder import Embedder
from pai_rag.data_ingestion.operators.parser import Parser
from pai_rag.data_ingestion.operators.split import Splitter
from pai_rag.data_ingestion.operators.writer import Writer
from pai_rag.data_ingestion.utils.concurrency_utils import compute_concurrency_count
from pai_rag.data_ingestion.utils.dataset_utils import get_input_files, get_input_files_with_es_backend

import sys
import math
import ray
import time
from loguru import logger

from pai_rag.data_ingestion.utils.filename_utils import BlockFileNameProvider
from pai_rag.data_ingestion.utils.vectordb_utils import get_vector_store


DEFAULT_WORKING_DIR = "/app"
DEFAULT_ROWS_PER_FILE = 10000


class RayExecutor:
    """
    Executor based on Ray.
    """
    def __init__(self, working_dir: str = DEFAULT_WORKING_DIR):
        self.working_dir = working_dir
        # init ray
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
        self.filename_provider = BlockFileNameProvider(run_label=self.execution_ts, file_format="jsonl")

    def _resolve_op_class(self, op_config: BaseOperatorConfig) -> BaseOperator:
        if isinstance(op_config, ParserConfig):
            return Parser
        elif isinstance(op_config, SplitterConfig):
            return Splitter
        elif isinstance(op_config, EmbedderConfig):
            return Embedder
        
        raise ValueError(f"Unknown operator config: {op_config}.")
        

    def run(self,
            op_configs: List[BaseOperatorConfig] = [],
            datasource_config: DataSourceConfig = None):
        """
        Running the dataset process pipeline.
        """
        start_time = time.time()

        if datasource_config is not None:
            datasource = FileDeltaDatasource(paths=datasource_config.input_path, file_extensions=datasource_config.file_extensions)
            dataset = ray.data.read_datasource(datasource)
            dataset.write_json(
                datasource_config.output_path,
                min_rows_per_file=DEFAULT_ROWS_PER_FILE,
                try_create_dir=True,
                filename_provider=self.filename_provider,
                force_ascii=False,
            )
        else:
            if len(op_configs) == 0:
                logger.info("No op_configs and datasource provided, skipping dataset process pipeline.")
                return
            dataset = ray.data.read_json(op_configs[0].input_path)

        for op_config in op_configs:
            OP_TYPE = self._resolve_op_class(op_config=op_config)
            op_concurrency = compute_concurrency_count(
                num_cpus=op_config.num_cpus,
                memory=op_config.memory,
                num_gpus=op_config.num_gpus,
            )
            if isinstance(op_config, EmbedderConfig):
                dataset = dataset.map_batches(
                    OP_TYPE,
                    batch_size=op_config.batch_size,
                    num_cpus=op_config.num_cpus,
                    num_gpus=op_config.num_gpus,
                    memory=op_config.memory,
                    concurrency=op_concurrency,
                    fn_constructor_kwargs={ "config": op_config }
                )
            else:
                dataset = dataset.flat_map(
                    OP_TYPE,
                    num_cpus=op_config.num_cpus,
                    num_gpus=op_config.num_gpus,
                    memory=op_config.memory,
                    concurrency=op_concurrency,
                    fn_constructor_kwargs={ "config": op_config }
                )

            # 保存op结果
            dataset.write_json(
                op_config.output_path,
                min_rows_per_file=DEFAULT_ROWS_PER_FILE,
                try_create_dir=True,
                filename_provider=self.filename_provider,
                force_ascii=False,
            )

        logger.info(f"All ops are done in {time.time() - start_time:.3f}s.")


ray_executor = RayExecutor()