"""Read PDF files."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from pai_rag.utils.markdown_utils import (
    transform_local_to_oss,
    is_horizontal_table,
)
from bs4 import BeautifulSoup
from llama_index.core import Settings

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
import magic_pdf.model as model_config
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from operator import itemgetter
import time
import tempfile
import re
from PIL import Image
import os
import json
from loguru import logger


model_config.__use_inside_model__ = True


IMAGE_MAX_PIXELS = 512 * 512
TABLE_SUMMARY_MAX_ROW_NUM = 5
TABLE_SUMMARY_MAX_COL_NUM = 10
TABLE_SUMMARY_MAX_CELL_TOKEN = 20
TABLE_SUMMARY_MAX_TOKEN = 200
PAGE_TABLE_SUMMARY_MAX_TOKEN = 400
IMAGE_URL_PATTERN = r"(https?://[^\s]+?[\s\w.-]*\.(jpg|jpeg|png|gif|bmp))"
IMAGE_LOCAL_PATTERN = r"!\[(?P<alt_text>.*?)\]\((?P<url>/[^()\s]+(?:\s[^()\s]*)?/\S*?\.(jpg|jpeg|png|gif|bmp))\)"
IMAGE_COMBINED_PATTERN = r"!\[.*?\]\((https?://[^\s()]+|/[^()\s]+(?:\s[^()\s]*)?/\S*?\.(jpg|jpeg|png|gif|bmp))\)"
DEFAULT_HEADING_DIFF_THRESHOLD = 2


class PaiPDFReader(BaseReader):
    """Read PDF files including texts, tables, images.

    Args:
        enable_table_summary (bool):  whether to use table_summary to process tables
    """

    def __init__(
        self,
        enable_mandatory_ocr: bool = False,
        enable_table_summary: bool = False,
        oss_cache: Any = None,
    ) -> None:
        self.enable_table_summary = enable_table_summary
        self.enable_mandatory_ocr = enable_mandatory_ocr
        self._oss_cache = oss_cache
        logger.info(
            f"PaiPdfReader created with enable_table_summary : {self.enable_table_summary}"
        )
        logger.info(
            f"PaiPdfReader created with enable_mandatory_ocr : {self.enable_mandatory_ocr}"
        )

    def _transform_local_to_oss(self, pdf_name: str, local_url: str):
        image = Image.open(local_url)
        return transform_local_to_oss(self._oss_cache, image, pdf_name)

    def replace_image_paths(self, pdf_name: str, content: str):
        local_image_pattern = IMAGE_LOCAL_PATTERN
        matches = re.findall(local_image_pattern, content)
        for alt_text, local_url, image_type in matches:
            if self._oss_cache:
                time_tag = int(time.time())
                oss_url = self._transform_local_to_oss(pdf_name, local_url)
                updated_alt_text = f"pai_rag_image_{time_tag}_{alt_text}"
                content = content.replace(
                    f"![{alt_text}]({local_url})", f"![{updated_alt_text}]({oss_url})"
                )
            else:
                content = content.replace(f"![{alt_text}]({local_url})", "")

        return content

    @staticmethod
    def perform_ocr(img_path: str) -> str:
        table_engine = RapidTable()
        ocr_engine = RapidOCR()
        ocr_result, _ = ocr_engine(img_path)
        table_html_str, table_cell_bboxes, elapse = table_engine(img_path, ocr_result)
        return table_html_str

    @staticmethod
    def html_table_to_list_of_lists(html):
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if not table:
            return []
        table_data = []
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            table_data.append([col.get_text(strip=True) for col in cols])
        return table_data

    @staticmethod
    def add_table_ocr_content(
        markdown_content: str, image_path: str, ocr_content: str
    ) -> str:
        pattern = rf"!\[(.*?)\]\({re.escape(image_path)}\)"
        regex = re.compile(pattern)
        replacement = "table"
        offset = 0  # 用于记录已插入文本导致的偏移量
        for match in regex.finditer(markdown_content):
            # 记录匹配的起始和结束偏移量
            start, end = match.span()
            # 提取alt_text的起始和结束位置，以便替换
            alt_start, alt_end = match.span(1)  # span(1)针对第一个捕获组
            new_alt_start = alt_start + offset
            new_alt_end = alt_end + offset

            markdown_content = (
                markdown_content[:new_alt_start]
                + replacement
                + markdown_content[new_alt_end:]
            )
            new_start = start + offset
            ocr_content = f"\n\n{ocr_content}\n\n"
            markdown_content = (
                markdown_content[:new_start]
                + ocr_content
                + markdown_content[new_start:]
            )
            # 更新偏移量，因为刚刚插入了新文本
            offset += len(ocr_content) + len(replacement)
        return markdown_content

    @staticmethod
    def limit_cell_size(cell: str, max_chars: int) -> str:
        return (cell[:max_chars] + "...") if len(cell) > max_chars else cell

    @staticmethod
    def limit_table_content(table: List[List]) -> List[List]:
        return [
            [
                PaiPDFReader.limit_cell_size(str(cell), TABLE_SUMMARY_MAX_CELL_TOKEN)
                for cell in row
            ]
            for row in table
        ]

    @staticmethod
    def tables_summarize(table: List[List]) -> str:
        table = PaiPDFReader.limit_table_content(table)
        if not is_horizontal_table(table):
            table = list(zip(*table))
        table = table[:TABLE_SUMMARY_MAX_ROW_NUM]
        table = [row[:TABLE_SUMMARY_MAX_COL_NUM] for row in table]

        prompt_text = f"请为以下表格生成一个摘要: {table}"
        response = Settings.llm.complete(
            prompt_text,
            max_tokens=200,
            n=1,
        )
        summarized_text = response
        return summarized_text.text

    def process_table(self, markdown_content, json_data):
        ocr_count = 0

        for item in json_data:
            if item["type"] == "table" and "img_path" in item:
                img_path = item["img_path"]
                if os.path.exists(img_path):
                    ocr_count += 1
                    ocr_content = PaiPDFReader.perform_ocr(img_path)
                    if self.enable_table_summary:
                        table_list_data = PaiPDFReader.html_table_to_list_of_lists(
                            ocr_content
                        )
                        summarized_table_text = PaiPDFReader.tables_summarize(
                            table_list_data
                        )[:TABLE_SUMMARY_MAX_TOKEN]
                        ocr_content += f"\n\n{summarized_table_text}\n\n"
                        markdown_content = PaiPDFReader.add_table_ocr_content(
                            markdown_content, item["img_path"], ocr_content
                        )
                    else:
                        markdown_content = PaiPDFReader.add_table_ocr_content(
                            markdown_content, item["img_path"], ocr_content
                        )
                else:
                    logger.warning(f"警告：图片文件不存在 {img_path}")
        return markdown_content

    def post_process_multi_level_headings(self, json_data, md_content):
        logger.info(
            "*****************************start process headings*****************************"
        )
        pages_list = json_data["pdf_info"]
        if not pages_list:
            return md_content
        text_height_min = float("inf")
        text_height_max = 0
        title_list = []
        for page in pages_list:
            page_infos = page["preproc_blocks"]
            for item in page_infos:
                if not item.get("lines", None) or len(item["lines"]) <= 0:
                    continue
                x0, y0, x1, y1 = item["lines"][0]["bbox"]
                content_height = y1 - y0
                if item["type"] == "title":
                    title_height = int(content_height)
                    title_text = ""
                    for line in item["lines"]:
                        for span in line["spans"]:
                            if span["type"] == "inline_equation":
                                span["content"] = " $" + span["content"] + "$ "
                            title_text += span["content"]
                    title_text = title_text.replace("\\", "\\\\")
                    title_list.append((title_text, title_height))
                elif item["type"] == "text":
                    if content_height < text_height_min:
                        text_height_min = content_height
                    if content_height > text_height_max:
                        text_height_max = content_height

        sorted_list = sorted(title_list, key=itemgetter(1), reverse=True)
        diff_list = [
            (sorted_list[i][1] - sorted_list[i + 1][1], i)
            for i in range(len(sorted_list) - 1)
        ]
        sorted_diff = sorted(diff_list, key=itemgetter(0), reverse=True)
        slice_index = []
        for diff, index in sorted_diff:
            # 标题差的绝对值超过2，则认为是下一级标题
            # markdown 中，# 表示一级标题，## 表示二级标题，以此类推，最多有6级标题，最多能有5次切分
            if diff >= DEFAULT_HEADING_DIFF_THRESHOLD and len(slice_index) <= 5:
                slice_index.append(index)
        slice_index.sort(reverse=True)
        rank = 1
        cur_index = 0
        if len(slice_index) > 0:
            cur_index = slice_index.pop()
        for index, (title_text, title_height) in enumerate(sorted_list):
            if index > cur_index:
                rank += 1
                if len(slice_index) > 0:
                    cur_index = slice_index.pop()
                else:
                    cur_index = len(sorted_list) - 1
            title_level = "#" * rank + " "
            if text_height_min <= text_height_max and int(
                text_height_min
            ) <= title_height <= int(text_height_max):
                title_level = ""
            old_title = "# " + title_text
            new_title = title_level + title_text
            md_content = re.sub(re.escape(old_title), new_title, md_content)

        return md_content

    def parse_pdf(
        self,
        pdf_path: str,
        parse_method: str = "auto",
        model_json_path: str = None,
    ):
        """
        执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到 pdf 文件所在的目录

        :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
        :param parse_method: 解析方法， 共 auto、ocr两种，默认 auto。auto会根据文件类型选择TXT模式或者OCR模式解析。ocr会直接使用OCR模式。
        :param model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
        """
        try:
            pdf_name = os.path.basename(pdf_path).split(".")[0]
            pdf_name = pdf_name.replace(" ", "_")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, pdf_name)
                pdf_bytes = open(pdf_path, "rb").read()  # 读取 pdf 文件的二进制数据

                if model_json_path:
                    model_json = json.loads(
                        open(model_json_path, "r", encoding="utf-8").read()
                    )
                else:
                    model_json = []

                # 执行解析步骤
                image_writer = FileBasedDataWriter(temp_file_path)

                # 选择解析方式
                if parse_method == "auto":
                    jso_useful_key = {"_pdf_type": "", "model_list": model_json}
                    pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
                elif parse_method == "ocr":
                    pipe = OCRPipe(pdf_bytes, model_json, image_writer)
                else:
                    logger.error("unknown parse method, only auto, ocr, txt allowed")
                    exit(1)

                # 执行分类
                pipe.pipe_classify()

                # 如果没有传入模型数据，则使用内置模型解析
                if len(model_json) == 0:
                    pipe.pipe_analyze()  # 解析

                # 执行解析
                pipe.pipe_parse()
                content_list = pipe.pipe_mk_uni_format(temp_file_path, drop_mode="none")
                md_content = pipe.pipe_mk_markdown(temp_file_path, drop_mode="none")
                md_content = self.post_process_multi_level_headings(
                    pipe.pdf_mid_data, md_content
                )
                md_content = self.process_table(md_content, content_list)
                new_md_content = self.replace_image_paths(pdf_name, md_content)

            return new_md_content

        except Exception as e:
            logger.error(e)
            return None

    def load_data(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format."""
        return self.load(file_path, metadata=metadata, extra_info=extra_info)

    def load(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): file path of PDF file (accepts string or Path).
            metadata (bool, optional): if metadata to be included or not. Defaults to True.
            extra_info (Optional[Dict], optional): extra information related to each document in dict format. Defaults to None.

        Raises:
            TypeError: if extra_info is not a dictionary.
            TypeError: if file_path is not a string or Path.

        Returns:
            List[Document]: list of documents.
        """
        if self.enable_mandatory_ocr:
            parse_method = "ocr"
        else:
            parse_method = "auto"
        md_content = self.parse_pdf(file_path, parse_method)
        logger.info(f"[PaiPDFReader] successfully processed pdf file {file_path}.")
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
            logger.info(f"processed pdf file {file_path} without metadata")
        logger.info(f"[PaiPDFReader] successfully loaded {len(docs)} nodes.")
        return docs
