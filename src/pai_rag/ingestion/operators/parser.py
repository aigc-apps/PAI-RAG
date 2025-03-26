import json
import time
from typing import List, Optional
from pai_rag.core.rag_module import resolve
from pai_rag.ingestion.operators.base import BaseOperator, OperatorName
from pai_rag.ingestion.utils.download_utils import download_models_via_lock
from pai_rag.ingestion.utils.formatters import convert_document_to_dict
from pai_rag.integrations.readers.pai.pai_data_reader import (
    BaseDataReaderConfig,
    PaiDataReader,
)
from pai_rag.utils.oss_client import OssClient
import ray
import fcntl
from loguru import logger


@ray.remote
class Parser(BaseOperator):
    def __init__(
        self,
        name: str = OperatorName.PARSER,
        batch_size: int = 10,
        device: str = "cpu",
        num_cpus: float = 1,
        num_gpus: Optional[float] = None,
        model_dir: str = None,
        output_filename: str = None,
        concat_csv_rows: bool = False,
        enable_mandatory_ocr: bool = False,
        format_sheet_data_to_json: bool = False,
        sheet_column_filters: List[str] = None,
        oss_bucket: str = None,
        oss_endpoint: str = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            batch_size=batch_size,
            device=device,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            model_dir=model_dir,
            output_filename=output_filename,
            **kwargs,
        )

        download_models_via_lock(self.model_dir, "PDF-Extract-Kit", self.accelerator)

        self.data_reader_config = BaseDataReaderConfig(
            concat_csv_rows=concat_csv_rows,
            enable_mandatory_ocr=enable_mandatory_ocr,
            format_sheet_data_to_json=format_sheet_data_to_json,
            sheet_column_filters=sheet_column_filters,
        )
        if oss_bucket is not None and oss_endpoint is not None:
            self.oss_store = resolve(
                cls=OssClient,
                bucket_name=oss_bucket,
                endpoint=oss_endpoint,
            )
        else:
            self.oss_store = None

        self.data_reader = resolve(
            cls=PaiDataReader,
            reader_config=self.data_reader_config,
            oss_store=self.oss_store,
        )
        logger.info(
            f"""Parser operator [PaiDataReader] init finished with following parameters:
                        concat_csv_rows: {concat_csv_rows}
                        enable_mandatory_ocr: {enable_mandatory_ocr}
                        format_sheet_data_to_json: {format_sheet_data_to_json}
                        sheet_column_filters: {sheet_column_filters}
                        oss_bucket: {oss_bucket}
                        oss_endpoint: {oss_endpoint}
            """
        )

    def process(self, input_files: List[str]) -> List[dict]:
        documents = self.data_reader.load_data(file_path_or_directory=input_files)
        if len(documents) == 0:
            logger.info(f"No data found in the input files: {input_files}")
            return []

        results = convert_document_to_dict(documents)
        self.persist(results)
        return results

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
