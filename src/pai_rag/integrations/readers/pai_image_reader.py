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
from llama_index.core.schema import Document, ImageDocument
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
import logging

logger = logging.getLogger(__name__)


class PaiImageReader(BaseReader):
    """Image parser.

    Args:
        multi-modal llm (LLM)

    """

    def __init__(
        self, multimodal_llm: Any, oss_cache: Any, *args: Any, **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._multimodal_llm = multimodal_llm
        self._oss_cache = oss_cache

    def load_image_urls(self, image_urls: List[str], extra_info: Optional[Dict] = None):
        logger.info(f"Get {len(image_urls)} image urls. Start parsing.")
        docs = []
        for url in tqdm(image_urls):
            image_documents = load_image_urls([url])
            image_response = self._multimodal_llm.complete(
                prompt="详细描述图片中是什么",
                image_documents=image_documents,
            )
            if image_response.text:
                print(image_response.text)
                metadata = {}
                url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
                metadata["file_name"] = f"{url_hash}_{os.path.basename(url)}"
                metadata["file_path"] = url
                metadata["image_url"] = url
                if extra_info:
                    metadata.update(extra_info)

                docs.append(Document(text=image_response.text, metadata=metadata))
            else:
                logger.warn(
                    f"Process {url} failed. Get empty response from multimodal model: {image_response.raw}"
                )
        logger.info(
            f"Finished processing {len(image_urls)} image urls into {len(docs)} docs."
        )
        return docs

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

            extra_info["file_path"] = file_path
            extra_info["file_name"] = os.path.basename(file_path)
            extra_info["image_url"] = image_url
            image_doc = ImageDocument(image_url=image_url, extra_info=extra_info)
            docs = [image_doc]
            # docs = self.load_image_urls([image_url], extra_info=extra_info)
        return docs
