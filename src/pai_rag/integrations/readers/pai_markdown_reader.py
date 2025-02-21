"""Read markdown files.

"""
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Union, Any
import re
import os
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pai_rag.utils.markdown_utils import transform_local_to_oss

from loguru import logger
from itertools import chain

REGEX_H1 = "===+"
REGEX_H2 = "---+"
REGEX_USELESS_PHRASE = "\{#[0-9a-z]+\}"  # Only for aliyun docs
MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\(((?!https?://|www\.)[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))\)",
    re.IGNORECASE,
)
HTML_IMAGE_PATTERN = re.compile(
    r'<img[^>]*src=["\']((?!https?://|www\.)[^"\']+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))["\'][^>]*>',
    re.IGNORECASE,
)


class PaiMarkdownReader(BaseReader):
    def __init__(
        self,
        oss_cache: Any = None,
    ) -> None:
        self._oss_cache = oss_cache

    def replace_image_paths(self, markdown_dir: str, markdown_name: str, content: str):
        markdown_image_matches = MARKDOWN_IMAGE_PATTERN.finditer(content)
        html_image_matches = HTML_IMAGE_PATTERN.finditer(content)
        all_image_matches = chain(markdown_image_matches, html_image_matches)
        for match in all_image_matches:
            full_match = match.group(0)  # 整个匹配
            local_url = match.group(1)  # 捕获的URL

            local_path = os.path.normpath(os.path.join(markdown_dir, local_url))

            if self._oss_cache:
                oss_url = self._transform_local_to_oss(markdown_name, local_path)
                if oss_url:
                    content = content.replace(local_url, oss_url)
                else:
                    content = content.replace(full_match, "")
            else:
                content = content.replace(full_match, "")

        return content

    def _transform_local_to_oss(self, markdown_name: str, local_url: str):
        try:
            image = Image.open(local_url)
            return transform_local_to_oss(self._oss_cache, image, markdown_name)
        except Exception as e:
            logger.error(f"read markdown local image failed: {e}")
            return None

    def parse_markdown(self, markdown_path):
        markdown_name = os.path.basename(markdown_path).split(".")[0]
        markdown_name = markdown_name.replace(" ", "_")
        markdown_dir = os.path.dirname(markdown_path)
        text = ""
        pre_line = ""
        with open(markdown_path) as fp:
            line = fp.readline()
            is_code = False
            while line:
                striped_line = re.sub(REGEX_USELESS_PHRASE, "", line)
                if striped_line.startswith("```"):
                    is_code = not is_code

                if not striped_line:
                    text += pre_line
                    pre_line = "\n"
                    line = fp.readline()
                elif re.match(REGEX_H1, striped_line):
                    text += f"# {pre_line}"
                    pre_line = ""
                    line = fp.readline()
                elif re.match(REGEX_H2, striped_line):
                    text += f"## {pre_line}"
                    pre_line = ""
                    line = fp.readline()
                else:
                    text += pre_line
                    pre_line = striped_line
                    if is_code or line.startswith("#") or line.endswith("  \n"):
                        pre_line = f"{striped_line}\n"
                    line = fp.readline()

        text += pre_line
        md_content = self.replace_image_paths(markdown_dir, markdown_name, text)
        return md_content

    def load_data(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from Markdown file and also accepts extra information in dict format."""
        return self.load(file_path, metadata=metadata, extra_info=extra_info)

    def load(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from Markdown file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): file path of Markdown file (accepts string or Path).
            metadata (bool, optional): if metadata to be included or not. Defaults to True.
            extra_info (Optional[Dict], optional): extra information related to each document in dict format. Defaults to None.

        Raises:
            TypeError: if extra_info is not a dictionary.
            TypeError: if file_path is not a string or Path.

        Returns:
            List[Document]: list of documents.
        """
        md_content = self.parse_markdown(file_path)

        logger.info(
            f"[PaiMarkdownReader] successfully processed markdown file {file_path}."
        )
        docs = []
        if metadata and extra_info:
            extra_info = extra_info
        else:
            extra_info = dict()
            logger.info(f"processed markdown file {file_path} without metadata")
        doc = Document(text=md_content, extra_info=extra_info)
        docs.append(doc)
        logger.info(f"[PaiMarkdownReader] successfully loaded {len(docs)} nodes.")
        return docs
