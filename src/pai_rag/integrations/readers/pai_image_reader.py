"""Tabular parser-CSV parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
import os
from tqdm import tqdm
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
import logging

logger = logging.getLogger(__name__)


class PaiImageReader(BaseReader):
    """Image parser.

    Args:
        multi-modal llm (LLM)

    """

    def __init__(self, multimodal_llm: Any, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._multimodal_llm = multimodal_llm

    def load_data(
        self,
        file_path: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        image_urls = []
        with open(file_path, "r", encoding="utf-8") as file:
            image_urls = [line.strip() for line in file.readlines()]

        logger.info(f"Get {len(image_urls)} image urls. Start parsing.")
        docs = []
        for url in tqdm(image_urls):
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()

            image_documents = load_image_urls([url])
            image_response = self._multimodal_llm.complete(
                prompt="详细描述图片中是什么",
                image_documents=image_documents,
            )
            print(url, image_response.text)

            metadata = {}
            metadata["file_name"] = f"{url_hash}_{os.path.basename(url)}"
            metadata["file_path"] = url
            metadata["image_url"] = url
            if extra_info:
                metadata.update(extra_info)

            docs.append(Document(text=image_response.text, metadata=metadata))
        return docs
