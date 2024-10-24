"""Html parser.

"""
import html2text
import logging
from bs4 import BeautifulSoup
import requests
from typing import Dict, List, Optional, Union, Any
from io import BytesIO
from pai_rag.utils.markdown_utils import (
    transform_local_to_oss,
    convert_table_to_markdown,
    PaiTable,
)
from pathlib import Path
import re
import time
import os
from PIL import Image
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

IMAGE_URL_PATTERN = (
    r"!\[(?P<alt_text>.*?)\]\((https?://[^\s]+?[\s\w.-]*\.(jpg|jpeg|png|gif|bmp))\)"
)


class PaiHtmlReader(BaseReader):
    """Read html files including texts, tables, images.

    Args:
        enable_table_summary (bool):  whether to use table_summary to process tables
    """

    def __init__(
        self,
        enable_table_summary: bool = False,
        oss_cache: Any = None,
    ) -> None:
        self.enable_table_summary = enable_table_summary
        self._oss_cache = oss_cache
        logger.info(
            f"PaiHtmlReader created with enable_table_summary : {self.enable_table_summary}"
        )

    def _extract_tables(self, html):
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            # 替换表格内容为一个占位符
            placeholder = f"<!-- TABLE_PLACEHOLDER_{id(table)} -->"
            table.replace_with(placeholder)
        return str(soup), tables

    def _get_table_dimension(self, table):
        max_rows = len(table.find_all("tr"))
        max_columns = 0
        for row in table.find_all("tr"):
            current_columns = 0
            for col in row.find_all(["th", "td"]):
                current_columns += 1
            max_columns = max(current_columns, max_columns)
        return max_rows, max_columns

    def _convert_table_to_list(self, table):
        max_rows, max_cols = self._get_table_dimension(table)
        table_matrix = [["" for _ in range(max_cols)] for _ in range(max_rows)]
        current_row_index = 0
        for row in table.find_all("tr"):
            current_col_index = 0
            for cell in row.find_all(["th", "td"]):
                cell_content = self._parse_cell_content(cell)
                col_span = int(cell.get("colspan", 1))
                row_span = int(cell.get("rowspan", 1))
                while (
                    current_col_index < max_rows
                    and table_matrix[current_row_index][current_col_index] != ""
                ):
                    current_col_index += 1
                if current_col_index >= max_cols or current_row_index >= max_rows:
                    break
                for i in range(col_span):
                    if current_col_index + i < max_cols:
                        table_matrix[current_row_index][
                            current_col_index + i
                        ] = cell_content
                for i in range(row_span):
                    if current_row_index + i < max_rows:
                        table_matrix[current_row_index + i][
                            current_col_index
                        ] = cell_content
                current_col_index += col_span
            current_row_index += 1

        return table_matrix, max_cols

    def _parse_cell_content(self, cell):
        content = []
        for element in cell.contents:
            images = element.find_all("img")
            image_links = [img.get("src") for img in images]
            for image_url in image_links:
                content.append(f"![]({image_url})")
            content.append(element.text)
        return " ".join(content)

    def _convert_table_to_markdown(self, table):
        table, total_cols = self._convert_table_to_list(table)
        headers = table[0]
        rows = table[1:]
        table = PaiTable(headers=[headers], rows=rows)
        return convert_table_to_markdown(table, total_cols)

    def _transform_local_to_oss(self, html_name: str, image_url: str):
        response = requests.get(image_url)
        response.raise_for_status()  # 检查请求是否成功

        # 将二进制数据转换为图像对象
        image = Image.open(BytesIO(response.content))
        return transform_local_to_oss(self._oss_cache, image, html_name)

    def _replace_image_paths(self, html_name: str, content: str):
        image_pattern = IMAGE_URL_PATTERN
        matches = re.findall(image_pattern, content)
        for alt_text, image_url, image_type in matches:
            time_tag = int(time.time())
            oss_url = self._transform_local_to_oss(html_name, image_url)
            updated_alt_text = f"pai_rag_image_{time_tag}_{alt_text}"
            content = content.replace(
                f"![{alt_text}]({image_url})", f"![{updated_alt_text}]({oss_url})"
            )

        return content

    def convert_html_to_markdown(self, html_path):
        html_name = os.path.basename(html_path).split(".")[0]
        html_name = html_name.replace(" ", "_")
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

            markdown_content = self._replace_image_paths(html_name, markdown_content)

            return markdown_content

        except Exception as e:
            logger(e)
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
        if metadata:
            if not extra_info:
                extra_info = {}
            doc = Document(text=md_content, extra_info=extra_info)

            docs.append(doc)
        else:
            doc = Document(
                text=md_content,
                extra_info=dict(),
            )
            docs.append(doc)
            logger.info(f"processed html file {file_path} without metadata")
        print(f"[PaiHtmlReader] successfully loaded {len(docs)} nodes.")
        return docs
