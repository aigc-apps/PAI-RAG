import os
import ray
import threading
from typing import List
from pathlib import Path
from urllib.parse import urlparse
from pai_rag.core.rag_module import resolve
from pai_rag.utils.oss_client import OssClient
from pai_rag.tools.data_process.ops.base_op import BaseOP, OPERATORS
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.integrations.readers.pai.pai_data_reader import BaseDataReaderConfig
from pai_rag.tools.data_process.utils.formatters import convert_document_to_dict
from pai_rag.tools.data_process.utils.download_utils import download_models_via_lock

OP_NAME = "rag_parser"


@OPERATORS.register_module(OP_NAME)
@ray.remote
class Parser(BaseOP):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    def __init__(
        self,
        concat_csv_rows: bool = False,
        enable_mandatory_ocr: bool = False,
        format_sheet_data_to_json: bool = False,
        sheet_column_filters: List[str] = None,
        oss_bucket: str = None,
        oss_endpoint: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
        self.mount_path = os.environ.get("INPUT_MOUNT_PATH", None)
        self.real_path = os.environ.get("OSS_SOURCE_PATH", None)
        if self.mount_path and self.real_path:
            self.mount_path = Path(self.mount_path.strip("/")).resolve()
            self.real_path = self.real_path.strip("/")
            real_uri = urlparse(self.real_path)
            if not real_uri.scheme:
                self.logger.error(
                    f"Real path '{self.real_path}' must include a URI scheme (e.g., 'oss://')."
                )
                self.should_replace = False
            else:
                self.should_replace = True
        else:
            self.should_replace = False
            self.logger.warning(
                "File path won't be replaced to data source URI since either INPUT_MOUNT_PATH or OSS_SOURCE_PATH is not provided."
            )
        self.logger.info(
            f"""ParserActor [PaiDataReader] init finished with following parameters:
                        concat_csv_rows: {concat_csv_rows}
                        enable_mandatory_ocr: {enable_mandatory_ocr}
                        format_sheet_data_to_json: {format_sheet_data_to_json}
                        sheet_column_filters: {sheet_column_filters}
                        oss_bucket: {oss_bucket}
                        oss_endpoint: {oss_endpoint}
                        path_should_replace: {self.should_replace}
            """
        )

    def replace_mount_with_real_path(self, documents):
        if self.should_replace:
            for document in documents:
                if "file_path" not in document.metadata:
                    continue
                file_path = document.metadata["file_path"]
                file_path_obj = Path(file_path).resolve()
                try:
                    relative_path_str = (
                        file_path_obj.relative_to(self.mount_path).as_posix().strip("/")
                    )
                    document.metadata[
                        "file_path"
                    ] = f"{self.real_path}/{relative_path_str}"
                    document.metadata["mount_path"] = file_path
                    self.logger.debug(
                        f"Replacing original file_path: {file_path} --> {document.metadata['file_path']}"
                    )
                except ValueError:
                    # file_path 不以 mount_path 开头
                    self.logger.debug(
                        f"Path {file_path} does not start with mount path {self.mount_path}. No replacement done."
                    )
                except Exception as e:
                    self.logger.error(f"Error replacing path {file_path}: {e}")

    def process(self, input_file):
        current_thread = threading.current_thread()
        self.logger.info(f"当前线程的 ID: {current_thread.ident} 进程ID: {os.getpid()}")
        documents = self.data_reader.load_data(file_path_or_directory=input_file)
        if len(documents) == 0:
            self.logger.info(f"No data found in the input file: {input_file}")
            return None
        self.replace_mount_with_real_path(documents)
        return convert_document_to_dict(documents)
