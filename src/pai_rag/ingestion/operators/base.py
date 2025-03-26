from enum import Enum
import fcntl
import json
import time
from typing import List, Optional
from pai_rag.ingestion.utils.cuda_utils import is_cuda_available
from loguru import logger


class OperatorName(str, Enum):
    PARSER = "rag_parser"
    SPLITTER = "rag_splitter"
    EMBEDDER = "rag_embedder"
    WRITER = "rag_writer"


class BaseOperator:
    def __init__(
        self,
        name: str = "default",
        batch_size: int = 10,
        device: str = "cpu",
        num_cpus: float = 1,
        num_gpus: Optional[float] = None,
        model_dir: str = None,
        output_filename: str = None,
        **kwargs,
    ):
        self.name = name
        self.batch_size = batch_size
        self.device = device
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.model_dir = model_dir
        self.output_filename = output_filename
        self.kwargs = kwargs

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def use_cuda(self):
        return self.device.lower() == "cuda" and is_cuda_available()

    def persist(self, results: List[dict]):
        logger.info(f"Start writing results to {self.output_filename}")

        with open(self.output_filename, "a") as file:
            start_time = time.time()
            lock_timeout = 3600
            while True:
                try:
                    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start_time > lock_timeout:
                        logger.warning(
                            f"Failed to acquire lock on {self.output_filename} after {lock_timeout} seconds"
                        )
                        raise TimeoutError(
                            f"Failed to acquire lock on {self.output_filename} after {lock_timeout} seconds"
                        )
                    logger.info("File is locked by another process, waiting...")
                    time.sleep(0.5)

            for result in results:
                json_line = json.dumps(result, ensure_ascii=False)
                file.write(f"{json_line}\n")
            fcntl.flock(file, fcntl.LOCK_UN)
