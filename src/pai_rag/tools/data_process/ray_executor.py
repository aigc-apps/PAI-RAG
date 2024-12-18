import os
import ray
import time
from loguru import logger
from pai_rag.tools.data_process.dataset.ray_dataset import RayDataset
from pai_rag.tools.data_process.dataset.file_dataset import FileDataset
from pai_rag.tools.data_process.utils.op_utils import (
    load_op_names,
    load_op,
    get_previous_operation,
)


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

    def run(self):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        op_names = load_op_names(self.cfg.process)
        all_tstart = time.time()
        for op_name in op_names:
            if op_name == "pai_rag_parser":
                dataset = FileDataset(self.cfg.dataset_path, self.cfg)
                self.cfg.dataset_path = self.cfg.export_path
            else:
                dataset = RayDataset(
                    os.path.join(
                        self.cfg.dataset_path, get_previous_operation(op_name)
                    ),
                    self.cfg,
                )
            op = load_op(op_name, self.cfg.process)
            logger.info("Processing op {op_name} ...")
            tstart = time.time()
            dataset.process(op, op_name)
            tend = time.time()
            logger.info(f"Op {op_name} is done in {tend - tstart:.3f}s.")
            ray.kill(op)

        all_tend = time.time()
        logger.info(f"All ops are done in {all_tend - all_tstart:.3f}s.")
