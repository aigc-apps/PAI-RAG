import os
from pathlib import Path
from typing import List

from pairag.data_pipeline.datasource.filedelta_datasource import FileDeltaDatasource
from pairag.data_pipeline.models.config.datasource import DataSourceConfig
from pairag.data_pipeline.models.config.operator import (
    BaseOperatorConfig,
    EmbedderConfig,
    ParserConfig,
    SplitterConfig,
    SinkConfig,
)
from pairag.data_pipeline.operators.base import BaseOperator
from pairag.data_pipeline.operators.embedder import Embedder
from pairag.data_pipeline.operators.parser import Parser
from pairag.data_pipeline.operators.split import Splitter
from pairag.data_pipeline.operators.sink import Sinker
from pairag.data_pipeline.utils.concurrency_utils import compute_concurrency_count
from pairag.data_pipeline.utils.dataset_utils import get_input_files

import ray
import time
from loguru import logger

from pairag.data_pipeline.utils.filename_utils import BlockFileNameProvider
from pairag.data_pipeline.utils.path_utils import clear_folder


DEFAULT_WORKING_DIR = "/app"
DEFAULT_ROWS_PER_FILE = 1000


class RayExecutor:
    """
    Executor based on Ray.
    """

    def __init__(self, working_dir: str = DEFAULT_WORKING_DIR):
        self.working_dir = working_dir
        # init ray
        print("model dir: ", os.environ.get("pairag_MODEL_DIR"))
        if os.environ.get("pairag_MODEL_DIR"):
            ray_env_model_dir = os.environ["pairag_MODEL_DIR"]
        else:
            ray_env_model_dir = os.path.join(self.working_dir, "model_repository")
            os.environ["pairag_MODEL_DIR"] = ray_env_model_dir
        logger.info(
            f"Initing Ray with working_dir: {self.working_dir}, set env: pairag_MODEL_DIR = {ray_env_model_dir}..."
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
        elif isinstance(op_config, SinkConfig):
            return Sinker

        raise ValueError(f"Unknown operator config: {op_config}.")

    def _need_batch_execution(self, op_config: BaseOperatorConfig) -> bool:
        return isinstance(op_config, EmbedderConfig) or isinstance(
            op_config, SinkConfig
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

        # 获取delta datasource
        if datasource_config is not None:
            clear_folder(datasource_config.output_path)
            Path(datasource_config.output_path).mkdir(parents=True, exist_ok=True)

            filename_provider = BlockFileNameProvider(
                run_label=f"read-{self.execution_ts}", file_format="jsonl"
            )

            datasource = FileDeltaDatasource(config=datasource_config)
            delta_dataset = ray.data.read_datasource(datasource).materialize()
            delta_dataset.write_json(
                datasource_config.output_path,
                min_rows_per_file=DEFAULT_ROWS_PER_FILE,
                try_create_dir=True,
                filename_provider=filename_provider,
                force_ascii=False,
            )
        elif len(op_configs) == 0:
            logger.warning("No op_configs provided, skipping dataset process pipeline.")
            return

        # 执行each op
        for op_config in op_configs:
            logger.info(f"Executing {op_config.name}...")
            clear_folder(op_config.output_path)
            Path(op_config.output_path).mkdir(parents=True, exist_ok=True)

            input_files_list = get_input_files(
                file_path_or_directory=op_config.input_path,
                filter_pattern="*.jsonl",
            )
            if len(input_files_list) == 0:
                logger.warning(f"No input files found for {op_config.name}.")
                return

            dataset = ray.data.read_json(input_files_list)

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
                )
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
                )

            # 保存op结果，保存向量库无需执行
            filename_provider = BlockFileNameProvider(
                run_label=f"{op_config.name.value}-{self.execution_ts}",
                file_format="jsonl",
            )

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
