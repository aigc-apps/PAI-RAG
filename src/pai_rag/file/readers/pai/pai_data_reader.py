from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
    before_sleep_log,
)
from pydantic import BaseModel
from typing import List, Any
import os
import pathlib
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document
from functools import partial
from pai_rag.file.readers.pai.constants import ACCEPTABLE_DOC_TYPES
from pai_rag.file.store.oss_store import PaiOssStore
import logging
from loguru import logger


class BaseDataReaderConfig(BaseModel):
    concat_csv_rows: bool = False
    enable_mandatory_ocr: bool = False
    format_sheet_data_to_json: bool = False
    sheet_column_filters: List[str] | None = None


def get_file_readers(reader_config: BaseDataReaderConfig = None, oss_store: Any = None):
    from pai_rag.file.readers.pai.file_readers.pai_excel_reader import (
        PaiPandasExcelReader,
    )
    from pai_rag.file.readers.pai.file_readers.pai_image_reader import PaiImageReader
    from pai_rag.file.readers.pai.file_readers.pai_pdf_reader import PaiPDFReader
    from pai_rag.file.readers.pai.file_readers.pai_html_reader import PaiHtmlReader
    from pai_rag.file.readers.pai.file_readers.pai_csv_reader import (
        PaiPandasCSVReader,
    )
    from pai_rag.file.readers.pai.file_readers.pai_jsonl_reader import PaiJsonLReader
    from pai_rag.file.readers.pai.file_readers.pai_docx_reader import PaiDocxReader
    from pai_rag.file.readers.pai.file_readers.pai_pptx_reader import PaiPptxReader
    from pai_rag.file.readers.pai.file_readers.pai_markdown_reader import (
        PaiMarkdownReader,
    )

    reader_config = reader_config or BaseDataReaderConfig()
    image_reader = PaiImageReader(oss_cache=oss_store)

    file_readers = {
        ".html": PaiHtmlReader(
            oss_cache=oss_store,  # Storing html images
        ),
        ".htm": PaiHtmlReader(
            oss_cache=oss_store,  # Storing html images
        ),
        ".docx": PaiDocxReader(
            oss_cache=oss_store,  # Storing docx images
        ),
        ".pdf": PaiPDFReader(
            enable_mandatory_ocr=reader_config.enable_mandatory_ocr,
            oss_cache=oss_store,  # Storing pdf images
        ),
        ".pptx": PaiPptxReader(
            oss_cache=oss_store,  # Storing pptx images
        ),
        ".md": PaiMarkdownReader(
            oss_cache=oss_store,  # Storing markdown images
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
        ".jsonl": PaiJsonLReader(),
        ".jpg": image_reader,
        ".jpeg": image_reader,
        ".png": image_reader,
    }

    return file_readers


def get_input_files(
    file_path_or_directory: str | List[str],
    filter_pattern: str = None,
):
    filter_pattern = filter_pattern or "*"

    input_files = []
    if isinstance(file_path_or_directory, list):
        # file list
        input_files = [
            f
            for f in file_path_or_directory
            if os.path.isfile(f)
            and pathlib.Path(f).suffix.lower() in ACCEPTABLE_DOC_TYPES
        ]
    elif isinstance(file_path_or_directory, str) and os.path.isdir(
        file_path_or_directory
    ):
        # glob from directory
        directory = pathlib.Path(file_path_or_directory)
        input_files = [
            f
            for f in directory.rglob(filter_pattern)
            if os.path.isfile(f)
            and pathlib.Path(f).suffix.lower() in ACCEPTABLE_DOC_TYPES
        ]
    elif pathlib.Path(file_path_or_directory).suffix.lower() in ACCEPTABLE_DOC_TYPES:
        # Single file
        input_files = [pathlib.Path(file_path_or_directory)]
    else:
        raise ValueError(
            f"Invalid input path or not supported file type for '{file_path_or_directory}'."
        )

    if not input_files:
        raise ValueError(
            f"No file found at path '{file_path_or_directory}' with pattern '{filter_pattern}'."
        )

    return input_files


def get_file_metadata(x, file_metadata_map):
    return file_metadata_map.get(x, {})


class PaiDataReader(BaseReader):
    def __init__(
        self,
        reader_config: BaseDataReaderConfig,
        oss_store: PaiOssStore = None,
    ):
        self.file_readers = get_file_readers(reader_config, oss_store)
        self.oss_store = oss_store

        logger.info(f"[PaiDataReader] created with {reader_config}")

    @retry(
        retry=retry_if_exception_type(OSError),
        wait=wait_fixed(2),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def load_data(
        self,
        file_path_or_directory=None,
        filter_pattern: str = None,
        from_oss: bool = False,
        oss_path: str = None,
        show_progress: bool = False,
    ) -> List[Document]:
        input_files = get_input_files(
            file_path_or_directory=file_path_or_directory,
            filter_pattern=filter_pattern,
        )
        file_metadata_map = {
            str(file): default_file_metadata_func(file_path=str(file))
            for file in input_files
        }

        file_metadata_func = partial(
            get_file_metadata, file_metadata_map=file_metadata_map
        )
        directory_reader = SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=self.file_readers,
            file_metadata=file_metadata_func,
            raise_on_error=True,
        )

        """Load data from the input directory."""

        try:
            documents = directory_reader.load_data(
                show_progress=show_progress,
            )
            return documents
        except OSError as e:
            logger.warning(f"读取{input_files}错误: {e}，重试中...")
            raise
        except Exception as e:
            logger.error(f"解析{input_files}错误: {e}")
            if e.__cause__:
                logger.error("解析错误原因: {e.__cause__}")
                raise e.__cause__
            else:
                raise

    async def aload_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return self.load_data(*args, **load_kwargs)
