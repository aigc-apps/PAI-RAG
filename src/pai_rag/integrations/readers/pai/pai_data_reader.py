from pydantic import BaseModel
from typing import List, Any
import os
from pai_rag.integrations.readers.markdown_reader import MarkdownReader
from pai_rag.integrations.readers.pai_image_reader import PaiImageReader
from pai_rag.integrations.readers.pai_pdf_reader import PaiPDFReader
from pai_rag.integrations.readers.html.html_reader import HtmlReader
from pai_rag.integrations.readers.pai_csv_reader import PaiPandasCSVReader
from pai_rag.integrations.readers.pai_excel_reader import PaiPandasExcelReader
from pai_rag.integrations.readers.pai_jsonl_reader import PaiJsonLReader

from llama_index.core.readers.base import BaseReader
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document
import logging

logger = logging.getLogger(__name__)

COMMON_FILE_PATH_FODER_NAME = "__pairag__knowledgebase__"


class BaseDataReaderConfig(BaseModel):
    concat_csv_rows: bool = False
    enable_table_summary: bool = False
    format_sheet_data_to_json: bool = False
    sheet_column_filters: List[str] = None


def get_file_readers(reader_config: BaseDataReaderConfig = None, oss_store: Any = None):
    reader_config = reader_config or BaseDataReaderConfig()
    image_reader = PaiImageReader(oss_cache=oss_store)

    file_readers = {
        ".html": HtmlReader(),
        ".htm": HtmlReader(),
        ".pdf": PaiPDFReader(
            enable_table_summary=reader_config.enable_table_summary,
            oss_cache=oss_store,  # Storing pdf images
        ),
        ".csv": PaiPandasCSVReader(
            concat_rows=reader_config.concat_csv_rows,
            format_sheet_data_to_json=reader_config.format_sheet_data_to_json,
            sheet_column_filters=reader_config.sheet_column_filters,
        ),
        ".xlsx": PaiPandasExcelReader(
            concat_rows=reader_config.concat_csv_rows,
            format_sheet_data_to_json=reader_config.format_sheet_data_to_json,
            sheet_column_filters=reader_config.sheet_column_filters,
        ),
        ".xls": PaiPandasExcelReader(
            concat_rows=reader_config.concat_csv_rows,
            format_sheet_data_to_json=reader_config.format_sheet_data_to_json,
            sheet_column_filters=reader_config.sheet_column_filters,
        ),
        ".md": MarkdownReader(),
        ".jsonl": PaiJsonLReader(),
        ".jpg": image_reader,
        ".jpeg": image_reader,
        ".png": image_reader,
    }

    return file_readers


def get_oss_files(oss_path: str, filter_pattern: str = None, oss_store: Any = None):
    files = []
    if oss_store:
        prefix = oss_store.parse_oss_prefix(oss_path)
        object_list = oss_store.list_objects(prefix)
        oss_file_path_dir = "localdata/oss_tmp"
        if not os.path.exists(oss_file_path_dir):
            os.makedirs(oss_file_path_dir)
        for oss_obj in object_list:
            if not oss_obj.key.endswith("/"):  # 不是目录
                logger.info(f"Downloading oss object: {oss_obj.key}")
                try:
                    set_public = oss_store.put_object_acl(oss_obj.key, "public-read")
                except Exception:
                    logger.error(f"Failed to set_public document {oss_obj.key}")
                if set_public:
                    save_filename = os.path.join(
                        oss_file_path_dir, COMMON_FILE_PATH_FODER_NAME, oss_obj.key
                    )
                    oss_store.get_object_to_file(
                        key=oss_obj.key, filename=save_filename
                    )
                    files.append(save_filename)
                    logger.info(f"Downloaded oss object: {oss_obj.key}")
                else:
                    logger.error(f"Failed to load document {oss_obj.key}")
    else:
        raise ValueError("OSS must be provided to upload file.")

    if not files:
        raise ValueError(f"No file found at OSS path '{oss_path}'.")
    print(files)
    return files


def get_input_files(
    file_path_or_directory: str | List[str],
    from_oss: bool = False,
    oss_path: str = None,
    filter_pattern: str = None,
    oss_store: Any = None,
):
    filter_pattern = filter_pattern or "*"
    if from_oss:
        return get_oss_files(
            oss_path=oss_path, filter_pattern=filter_pattern, oss_store=oss_store
        )

    if isinstance(file_path_or_directory, list):
        # file list
        input_files = [f for f in file_path_or_directory if os.path.isfile(f)]
    elif isinstance(file_path_or_directory, str) and os.path.isdir(
        file_path_or_directory
    ):
        # glob from directory
        import pathlib

        directory = pathlib.Path(file_path_or_directory)
        input_files = [f for f in directory.rglob(filter_pattern) if os.path.isfile(f)]
    else:
        # Single file
        input_files = [file_path_or_directory]

    if not input_files:
        raise ValueError(
            f"No file found at path '{file_path_or_directory}' with pattern '{filter_pattern}'."
        )
    return input_files


class PaiDataReader(BaseReader):
    def __init__(
        self,
        reader_config: BaseDataReaderConfig,
        oss_store: Any = None,
    ):
        self.file_readers = get_file_readers(reader_config, oss_store)
        self.oss_store = oss_store

    def load_data(
        self,
        file_path_or_directory: str | List[str],
        filter_pattern: str = None,
        from_oss: bool = False,
        oss_path: str = None,
        show_progress: bool = False,
    ) -> List[Document]:
        input_files = get_input_files(
            file_path_or_directory=file_path_or_directory,
            from_oss=from_oss,
            oss_path=oss_path,
            filter_pattern=filter_pattern,
            oss_store=self.oss_store,
        )
        directory_reader = SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=self.file_readers,
        )

        """Load data from the input directory."""
        return directory_reader.load_data(show_progress=show_progress)

    async def aload_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return self.load_data(*args, **load_kwargs)
