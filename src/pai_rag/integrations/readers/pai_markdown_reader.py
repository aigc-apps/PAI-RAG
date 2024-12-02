"""Read markdown files.

"""
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Union, Any
import re
import time
import os
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pai_rag.utils.markdown_utils import transform_local_to_oss

from loguru import logger

REGEX_H1 = "===+"
REGEX_H2 = "---+"
REGEX_USELESS_PHRASE = "\{#[0-9a-z]+\}"  # Only for aliyun docs
IMAGE_URL_PATTERN = r"!\[(?P<alt_text>.*?)\]\((?P<url>(?:https?://[^\s()]+|[^\s()]+)\.(?P<image_type>jpg|jpeg|png|gif|bmp))\)"


class PaiMarkdownReader(BaseReader):
    def __init__(
        self,
        enable_table_summary: bool = False,
        oss_cache: Any = None,
    ) -> None:
        self.enable_table_summary = enable_table_summary
        self._oss_cache = oss_cache
        logger.info(
            f"PaiMarkdownReader created with enable_table_summary : {self.enable_table_summary}"
        )

    def replace_image_paths(self, markdown_name: str, content: str):
        image_pattern = IMAGE_URL_PATTERN
        matches = re.findall(image_pattern, content)
        for alt_text, local_url, image_type in matches:
            if self._oss_cache:
                time_tag = int(time.time())
                oss_url = self._transform_local_to_oss(markdown_name, local_url)
                updated_alt_text = f"pai_rag_image_{time_tag}_{alt_text}"
                if oss_url:
                    content = content.replace(
                        f"![{alt_text}]({local_url})",
                        f"![{updated_alt_text}]({oss_url})",
                    )
            else:
                content = content.replace(f"![{alt_text}]({local_url})", "")

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
        md_content = self.replace_image_paths(markdown_name, text)
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
