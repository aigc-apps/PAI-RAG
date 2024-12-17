import ray
import json
import os
import time
from loguru import logger
from pai_rag.tools.data_process.dataset.base_dataset import BaseDataset
from pai_rag.integrations.readers.pai.pai_data_reader import get_input_files


class FileDataset(BaseDataset):
    def __init__(self, dataset_path: str = None, cfg=None) -> None:
        self.data = get_input_files(dataset_path)
        if cfg:
            self.export_path = cfg.export_path

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
            run_tasks = [op.process.remote(input_file) for input_file in self.data]
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
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + "\n")
        logger.info(f"Exported dataset to {export_file_path} completed.")
