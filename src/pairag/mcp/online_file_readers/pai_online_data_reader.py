from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
    before_sleep_log,
)
from typing import Dict, List, Any
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.core.readers.base import BasePydanticReader, BaseReader
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.core.bridge.pydantic import Field
from functools import partial
from pairag.file.readers.pai.constants import ONLINE_ACCEPTABLE_DOC_TYPES
import logging
from loguru import logger

from pairag.file.store.pai_image_store import PaiImageStore
from pairag.file.readers.pai.pai_data_reader import get_input_files, get_file_metadata


def get_online_file_readers(
    image_store: PaiImageStore = None,
):
    from pairag.mcp.online_file_readers.pai_online_pdf_reader import PaiOnlinePDFReader
    from pairag.file.readers.pai.file_readers.pai_docx_reader import PaiDocxReader
    from pairag.file.readers.pai.file_readers.pai_markdown_reader import (
        PaiMarkdownReader,
    )

    file_readers = {
        ".docx": PaiDocxReader(
            image_store=image_store,  # Storing docx images
        ),
        ".pdf": PaiOnlinePDFReader(
            image_store=image_store,  # Storing pdf images
        ),
        ".md": PaiMarkdownReader(
            image_store=image_store,  # Storing markdown images
        ),
    }

    return file_readers


class PaiOnlineDataReader(BasePydanticReader):
    image_store: PaiImageStore = Field(default=None)
    file_readers: Dict[str, BaseReader] = Field(default={})

    def __init__(
        self,
        image_store: PaiImageStore = None,
    ):
        super().__init__()
        self.file_readers = get_online_file_readers(image_store)
        self.image_store = image_store

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
        supported_file_types: List[str] = ONLINE_ACCEPTABLE_DOC_TYPES,
        show_progress: bool = False,
    ) -> List[Document]:
        input_files = get_input_files(
            file_path_or_directory=file_path_or_directory,
            filter_pattern=filter_pattern,
            supported_file_types=supported_file_types,
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
                logger.error(f"解析错误原因: {e.__cause__}")
                raise e.__cause__
            else:
                raise

    async def aload_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return self.load_data(*args, **load_kwargs)
