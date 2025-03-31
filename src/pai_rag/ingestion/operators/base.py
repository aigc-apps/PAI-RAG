from enum import Enum
import fcntl
import json
import math
import time
import os
from typing import List, Optional
from pai_rag.ingestion.utils.cuda_utils import is_cuda_available
from loguru import logger


class OperatorName(str, Enum):
    PARSER = "rag_parser"
    SPLITTER = "rag_splitter"
    EMBEDDER = "rag_embedder"
    WRITER = "rag_writer"


OUTPUT_BATCH_SIZE = 50000


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
        self.result_count = 0
        self.real_output_filename = self._get_output_filename()

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def use_cuda(self):
        return self.device.lower() == "cuda" and is_cuda_available()


    def _get_output_filename(self):
        if self.output_filename is None:
            raise ValueError("output_filename must be specified")
        
        file_prefix, file_suffix = os.path.splitext(self.output_filename)
        file_idx = math.floor(self.result_count / OUTPUT_BATCH_SIZE) + 1
        return f"{file_prefix}_{file_idx:05d}{file_suffix}"
    
    def persist(self, results: List[dict]):
        logger.info(f"Start writing results to {self.output_filename}")
        
        output_dir = os.path.dirname(self.output_filename)
        os.makedirs(output_dir, exist_ok=True)
        self.result_count += len(results)

        print(f"creating dir {output_dir}")

        with open(self.real_output_filename, "a") as file:
            for result in results:
                json_line = json.dumps(result, ensure_ascii=False)
                file.write(f"{json_line}\n")

        logger.info(f"Finished writing {self.name} results to {self.real_output_filename}. Current process count: {self.result_count}")
        self.real_output_filename = self._get_output_filename()
