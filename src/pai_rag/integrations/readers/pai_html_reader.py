"""Html parser.

"""
import html2text
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, Any
from pai_rag.utils.markdown_utils import (
    transform_local_to_oss,
    convert_table_to_markdown,
    PaiTable,
)
from pathlib import Path
import re
import os
from PIL import Image
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from loguru import logger
from itertools import chain


MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\(((?!https?://|www\.)[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))\)",
    re.IGNORECASE,
)
HTML_IMAGE_PATTERN = re.compile(
    r'<img[^>]*src=["\']((?!https?://|www\.)[^"\']+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))["\'][^>]*>',
    re.IGNORECASE,
)


class PaiHtmlReader(BaseReader):
    """Read html files including texts, tables, images.

    Args:
        oss_cache :  oss_cache
    """

    def __init__(
        self,
        oss_cache: Any = None,
    ) -> None:
        self._oss_cache = oss_cache

    def _extract_tables(self, html):
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            # 替换表格内容为一个占位符
            placeholder = f"<!-- TABLE_PLACEHOLDER_{id(table)} -->"
            table.replace_with(placeholder)
        return str(soup), tables

    def _convert_table_to_pai_table(self, table):
        # 标记header的index
        row_headers_index = []
        col_headers_index = []
        row_header_flag = True
        col_header_index_max = -1
        table_matrix = []
        current_row_index = 0
        max_cols = 0
        max_rows = 0
        for row in table.find_all("tr"):
            current_col_index = 0
            if current_row_index == 0:
                row_cells = []
            else:
                row_cells = [""] * max_cols
            if current_row_index >= max_rows:
                table_matrix.append(row_cells)
                max_rows += 1
            for cell in row.find_all(["th", "td"]):
                if cell.name != "th":
                    row_header_flag = False
                elif cell.name == "th" and current_row_index != 0:
                    col_header_index_max = max(col_header_index_max, current_col_index)
                cell_content = self._parse_cell_content(cell)
                col_span = int(cell.get("colspan", 1))
                row_span = int(cell.get("rowspan", 1))
                if current_row_index != 0:
                    while (
                        current_col_index < max_cols
                        and table_matrix[current_row_index][current_col_index] != ""
                    ):
                        current_col_index += 1
                if (current_col_index > max_cols and max_cols != 0) or (
                    current_row_index > max_rows and max_rows != 0
                ):
                    break
                for i in range(col_span):
                    if current_row_index == 0:
                        table_matrix[current_row_index].append(cell_content)
                    elif current_col_index + i < max_cols:
                        table_matrix[current_row_index][
                            current_col_index + i
                        ] = cell_content

                if current_row_index == 0:
                    max_cols += col_span
                for i in range(1, row_span):
                    if current_row_index + i >= max_rows:
                        row_cells = [""] * max_cols
                        table_matrix.append(row_cells)
                        max_rows += 1
                    table_matrix[current_row_index + i][
                        current_col_index
                    ] = cell_content
                max_rows = max(current_row_index + row_span, max_rows)
                current_col_index += col_span
            if row_header_flag:
                row_headers_index.append(current_row_index)
            current_row_index += 1

        for i in range(col_header_index_max + 1):
            col_headers_index.append(i)

        table = PaiTable(
            data=table_matrix,
            row_headers_index=row_headers_index,
            column_headers_index=col_headers_index,
        )

        return table, max_cols

    def _parse_cell_content(self, cell):
        content = []
        for element in cell.contents:
            if isinstance(element, str):
                content.append(element.strip())
            elif element.name == "p":
                p_content = []
                for sub_element in element.contents:
                    if sub_element.name == "img":
                        image_url = sub_element.get("src")
                        p_content.append(f"![]({image_url})")
                    elif isinstance(sub_element, str):
                        p_content.append(sub_element.strip())
                    else:
                        p_content.append(sub_element.text.strip())
                content.append(" ".join(p_content))
            else:
                content.append(element.text.strip())
        return " ".join(content)

    def _convert_table_to_markdown(self, table):
        table, total_cols = self._convert_table_to_pai_table(table)
        return convert_table_to_markdown(table, total_cols)

    def _transform_local_to_oss(self, html_name: str, local_url: str):
        try:
            image = Image.open(local_url)
            return transform_local_to_oss(self._oss_cache, image, html_name)
        except Exception as e:
            logger.error(f"read html local image failed: {e}")
            return None

    def _replace_image_paths(self, html_dir: str, html_name: str, content: str):
        markdown_image_matches = MARKDOWN_IMAGE_PATTERN.finditer(content)
        html_image_matches = HTML_IMAGE_PATTERN.finditer(content)
        all_image_matches = chain(markdown_image_matches, html_image_matches)
        for match in all_image_matches:
            full_match = match.group(0)  # 整个匹配
            local_url = match.group(1)  # 捕获的URL

            local_path = os.path.normpath(os.path.join(html_dir, local_url))

            if self._oss_cache:
                oss_url = self._transform_local_to_oss(html_name, local_path)
                if oss_url:
                    content = content.replace(local_url, oss_url)
                else:
                    content = content.replace(full_match, "")
            else:
                content = content.replace(full_match, "")

        return content

    def convert_html_to_markdown(self, html_path):
        html_name = os.path.basename(html_path).split(".")[0]
        html_name = html_name.replace(" ", "_")
        html_dir = os.path.dirname(html_path)
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            modified_html, tables = self._extract_tables(html_content)
            h = html2text.HTML2Text()

            # 配置 html2text 对象
            h.ignore_links = True  # 是否忽略链接
            h.ignore_images = False  # 是否忽略图片
            h.escape_all = True  # 是否转义所有特殊字符
            h.body_width = 0  # 设置行宽为 0 表示不限制行宽

            # 将 HTML 转换为 Markdown
            markdown_content = h.handle(modified_html)
            for table in tables:
                table_markdown = self._convert_table_to_markdown(table) + "\n\n"
                placeholder = f"<!-- TABLE_PLACEHOLDER_{id(table)} -->"
                markdown_content = markdown_content.replace(placeholder, table_markdown)
            markdown_content = self._replace_image_paths(
                html_dir, html_name, markdown_content
            )

            return markdown_content

        except Exception as e:
            logger.exception(e)
            return None

    def load_data(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from Html file and also accepts extra information in dict format."""
        return self.load(file_path, metadata=metadata, extra_info=extra_info)

    def load(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from Html file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): file path of Html file (accepts string or Path).
            metadata (bool, optional): if metadata to be included or not. Defaults to True.
            extra_info (Optional[Dict], optional): extra information related to each document in dict format. Defaults to None.

        Raises:
            TypeError: if extra_info is not a dictionary.
            TypeError: if file_path is not a string or Path.

        Returns:
            List[Document]: list of documents.
        """

        md_content = self.convert_html_to_markdown(file_path)
        logger.info(f"[PaiHtmlReader] successfully processed html file {file_path}.")
        docs = []
        if metadata and extra_info:
            extra_info = extra_info
        else:
            extra_info = dict()
            logger.info(f"processed html file {file_path} without metadata")
        doc = Document(text=md_content, extra_info=extra_info)
        docs.append(doc)
        logger.info(f"[PaiHtmlReader] successfully loaded {len(docs)} nodes.")
        return docs
