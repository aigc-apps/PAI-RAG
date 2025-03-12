from pydantic import BaseModel
from typing import List, Any
import os
import pathlib
from pai_rag.integrations.readers.pai_image_reader import PaiImageReader
from pai_rag.integrations.readers.pai_pdf_reader import PaiPDFReader
from pai_rag.integrations.readers.pai_html_reader import PaiHtmlReader
from pai_rag.integrations.readers.pai_csv_reader import (
    PaiExcelReader,
    PaiPandasCSVReader,
)
from pai_rag.integrations.readers.pai_jsonl_reader import PaiJsonLReader
from pai_rag.integrations.readers.pai_docx_reader import PaiDocxReader
from pai_rag.integrations.readers.pai_pptx_reader import PaiPptxReader
from pai_rag.integrations.readers.pai_markdown_reader import PaiMarkdownReader
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document
from functools import partial
from pai_rag.integrations.readers.pai.constants import ACCEPTABLE_DOC_TYPES
from loguru import logger


COMMON_FILE_PATH_FODER_NAME = "__pairag__knowledgebase__"


class BaseDataReaderConfig(BaseModel):
    concat_csv_rows: bool = False
    enable_mandatory_ocr: bool = False
    format_sheet_data_to_json: bool = False
    sheet_column_filters: List[str] | None = None
    number_workers: int = 4


def get_file_readers(reader_config: BaseDataReaderConfig = None, oss_store: Any = None):
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
        ".xlsx": PaiExcelReader(oss_cache=oss_store),
        ".xls": PaiExcelReader(oss_cache=oss_store),
        ".jsonl": PaiJsonLReader(),
        ".jpg": image_reader,
        ".jpeg": image_reader,
        ".png": image_reader,
    }

    return file_readers


def get_oss_files(oss_path: str, filter_pattern: str = None, oss_store: Any = None):
    files = []
    file_metadata_map = {}
    if oss_store:
        prefix = oss_store.parse_oss_prefix(oss_path)
        object_list = oss_store.list_objects(prefix)
        oss_file_path_dir = "localdata/oss_tmp"
        if not os.path.exists(oss_file_path_dir):
            os.makedirs(oss_file_path_dir)
        for oss_obj in object_list:
            if (
                not oss_obj.key.endswith("/")
                and pathlib.Path(oss_obj.key).suffix.lower() in ACCEPTABLE_DOC_TYPES
            ):  # 不是目录
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
                    file_metadata_map[save_filename] = default_file_metadata_func(
                        file_path=save_filename
                    )
                    file_metadata_map[save_filename][
                        "file_url"
                    ] = oss_store.get_obj_key_url(oss_obj.key)
                    logger.info(f"Downloaded oss object: {oss_obj.key}")
                else:
                    logger.error(f"Failed to load document {oss_obj.key}")
    else:
        raise ValueError("OSS must be provided to upload file.")

    if not files:
        raise ValueError(f"No file found at OSS path '{oss_path}'.")
    return files, file_metadata_map


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

    if not input_files:
        raise ValueError(
            f"No file found at path '{file_path_or_directory}' with pattern '{filter_pattern}'."
        )

    file_metadata_map = {
        str(file): default_file_metadata_func(file_path=str(file))
        for file in input_files
    }
    return input_files, file_metadata_map


def get_file_metadata(x, file_metadata_map):
    return file_metadata_map.get(x, {})


class PaiDataReader(BaseReader):
    def __init__(
        self,
        reader_config: BaseDataReaderConfig,
        oss_store: Any = None,
    ):
        self.file_readers = get_file_readers(reader_config, oss_store)
        self.number_workers = reader_config.number_workers
        self.oss_store = oss_store

        logger.info(
            f"[PaiDataReader] created with number_workers : {self.number_workers}"
        )

    def load_data(
        self,
        file_path_or_directory=None,
        filter_pattern: str = None,
        from_oss: bool = False,
        oss_path: str = None,
        show_progress: bool = False,
    ) -> List[Document]:
        input_files, file_metadata_map = get_input_files(
            file_path_or_directory=file_path_or_directory,
            from_oss=from_oss,
            oss_path=oss_path,
            filter_pattern=filter_pattern,
            oss_store=self.oss_store,
        )

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
        documents = directory_reader.load_data(
            show_progress=show_progress,
            num_workers=self.number_workers,
        )
        return documents

    async def aload_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return self.load_data(*args, **load_kwargs)
