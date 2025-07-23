"""Tabular parser-CSV parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
import os
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument

from pairag.file.readers.pai.utils.image_utils import image_from_url
from pairag.file.store.pai_image_store import PaiImageStore


class PaiImageReader(BaseReader):
    """Image parser.

    Args:
        multi-modal llm (LLM)

    """

    def __init__(self, image_store: PaiImageStore, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.image_store = image_store

    def load_data(
        self,
        file_path: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        if self.image_store is None:
            raise Exception(
                f"Oss config must be provided for image processing for file {file_path}."
            )

        file_name = os.path.basename(file_path)
        image_url = self.image_store.upload_image(
            image_from_url(file_path), doc_name="image_docs"
        )
        if image_url is None:
            return []

        if extra_info is None:
            extra_info = {}
        extra_info["file_path"] = str(file_path)
        extra_info["file_name"] = file_name
        extra_info["image_url"] = image_url
        image_doc = ImageDocument(image_url=image_url, extra_info=extra_info)

        docs = [image_doc]
        return docs
