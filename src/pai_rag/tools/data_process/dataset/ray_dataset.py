import ray
from pai_rag.tools.data_process.dataset.base_dataset import BaseDataset
from loguru import logger
from pai_rag.tools.data_process.utils.process_utils import calculate_np
from pai_rag.tools.data_process.utils.cuda_util import get_num_gpus
from ray.data.datasource.filename_provider import _DefaultFilenameProvider
import time


class RayDataset(BaseDataset):
    def __init__(self, dataset_path: str = None, cfg=None) -> None:
        self.data = ray.data.read_json(dataset_path)
        self.export_path = self.cfg.export_path
        self.num_proc = None
        if cfg:
            self.num_proc = cfg.np

    def process(self, operators):
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]
        for op in operators:
            self._run_single_op(op)
            self.write_json(f"{self.export_path}/{op._name}")
        return self

    def _run_single_op(self, op):
        op_proc = calculate_np(
            op._name, op.mem_required, op.cpu_required, self.num_proc, op.use_cuda()
        )
        num_gpus = get_num_gpus(op, op_proc)
        try:
            batch_size = getattr(op, "batch_size", 1) if op.is_batched_op() else 1
            self.data = self.data.map_batches(
                op.process,
                batch_size=batch_size,
                batch_format="pyarrow",
                num_gpus=num_gpus,
            )
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op._name}].")
            import traceback

            traceback.print_exc()
            exit(1)

    def write_json(self, export_path):
        self.data = self.data.repartition(1)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.data.write_json(
            export_path,
            filename_provider=_DefaultFilenameProvider(
                dataset_uuid=timestamp, file_format="jsonl"
            ),
            force_ascii=False,
        )
