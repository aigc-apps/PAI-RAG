"""Tabular parser-CSV parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
import os
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument
import logging

logger = logging.getLogger(__name__)


class PaiImageReader(BaseReader):
    """Image parser.

    Args:
        multi-modal llm (LLM)

    """

    def __init__(self, oss_cache: Any, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._oss_cache = oss_cache

    def load_data(
        self,
        file_path: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        assert (
            self._oss_cache is not None
        ), "Oss config must be provided for image processing."

        file_ext = os.path.splitext(file_path)[1]
        with open(file_path, "rb") as file:
            data = file.read()
            image_url = self._oss_cache.put_object_if_not_exists(
                data=data,
                file_ext=file_ext,
                headers={
                    "x-oss-object-acl": "public-read"
                },  # set public read to make image accessible
                path_prefix="pairag/images/",
            )

            extra_info["file_path"] = str(file_path)
            extra_info["file_name"] = os.path.basename(file_path)
            extra_info["image_url"] = image_url
            image_doc = ImageDocument(image_url=image_url, extra_info=extra_info)
            docs = [image_doc]
            # docs = self.load_image_urls([image_url], extra_info=extra_info)
        return docs
