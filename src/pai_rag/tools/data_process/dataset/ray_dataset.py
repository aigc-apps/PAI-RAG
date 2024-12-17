import os
import ray
import time
import json
from pathlib import Path
from loguru import logger
from pai_rag.tools.data_process.utils.formatters import NumpyEncoder
from pai_rag.tools.data_process.dataset.base_dataset import BaseDataset


class RayDataset(BaseDataset):
    def __init__(self, dataset_path: str = None, cfg=None) -> None:
        self.batch_size = 10
        if os.path.isfile(dataset_path):
            self.data = ray.data.read_json(dataset_path)
        else:
            files = [
                str(file) for file in Path(dataset_path).rglob("*") if file.is_file()
            ]
            self.data = self.read_jsonl_in_batches(files)
        self.num_proc = None
        if cfg:
            self.export_path = cfg.export_path

    def read_jsonl_in_batches(self, files):
        for file_path in files:
            with open(file_path, "r") as file:
                batch = []
                for line in file:
                    batch.append(json.loads(line))
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

    def process(self, operators):
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]
        for op in operators:
            self._run_single_op(op)
            self.write_json(status=ray.get(op.get_name.remote()))
        return self

    def _run_single_op(self, op):
        try:
            logger.info(f"Running Op [{ray.get(op.get_name.remote())}].")
            run_tasks = [op.process.remote(batch_data) for batch_data in self.data]
            self.data = ray.get(run_tasks)
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op._name}].")
            import traceback

            traceback.print_exc()
            exit(1)

    def write_json(self, status):
        logger.info("Exporting parser dataset to disk...")
        export_path = os.path.join(self.export_path, status)
        os.makedirs(export_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        export_file_path = os.path.join(export_path, f"results_{timestamp}.jsonl")
        with open(export_file_path, "w") as f:
            for result in self.data:
                for line in result:
                    json_line = json.dumps(line, ensure_ascii=False, cls=NumpyEncoder)
                    f.write(json_line + "\n")
        logger.info(f"Exported dataset to {export_file_path} completed.")
