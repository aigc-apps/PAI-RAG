"""Read PDF files."""

from pathlib import Path
from typing import Dict, List, Optional, Union, TypedDict, Any
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import PyPDF2
from PyPDF2 import PageObject
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTRect, LTFigure, LTTextBoxHorizontal, LTTextLineHorizontal
import pdfplumber
from pdf2image import convert_from_path
import easyocr
from llama_index.core import Settings
from pai_rag.utils.constants import DEFAULT_EASYOCR_MODEL_DIR
import json
import sys
import unicodedata
import logging
import tempfile

logger = logging.getLogger(__name__)


class PageItem(TypedDict):
    page_number: int
    index_id: int
    item_type: str
    element: Any
    table_num: int
    text: str


class PaiPDFReader(BaseReader):
    """Read PDF files including texts, tables, images.
    Args:
    enable_image_ocr (bool): whether load ocr model to process images
    model_dir: (str): ocr model path
    """

    def __init__(
        self, enable_image_ocr: bool = False, model_dir: str = DEFAULT_EASYOCR_MODEL_DIR
    ) -> None:
        self.enable_image_ocr = enable_image_ocr
        if self.enable_image_ocr:
            self.model_dir = model_dir or DEFAULT_EASYOCR_MODEL_DIR
            logger.info("start loading ocr model")
            self.image_reader = easyocr.Reader(
                ["ch_sim", "en"],
                model_storage_directory=self.model_dir,
                download_enabled=True,
                detector=True,
                recognizer=True,
            )
            logger.info("finished loading ocr model")

    """剪切图片
        """

    def process_image(self, element: LTFigure, page_object: PageObject) -> str:
        # 获取从PDF中裁剪图像的坐标
        [image_left, image_top, image_right, image_bottom] = [
            element.x0,
            element.y0,
            element.x1,
            element.y1,
        ]
        # 使用坐标(left, bottom, right, top)裁剪页面
        page_object.mediabox.lower_left = (image_left, image_bottom)
        page_object.mediabox.upper_right = (image_right, image_top)
        # 将裁剪后的页面保存为新的PDF
        cropped_pdf_writer = PyPDF2.PdfWriter()
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".pdf"
        ) as cropped_pdf_file:
            cropped_pdf_writer.add_page(page_object)
            cropped_pdf_writer.write(cropped_pdf_file)
            cropped_pdf_file.flush()
            return self.convert_to_images(cropped_pdf_file.name)

    """创建一个将PDF内容转换为image的函数
    """

    def convert_to_images(self, input_file: str) -> str:
        images = convert_from_path(input_file)
        image = images[0]
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".png"
        ) as output_image_file:
            image.save(output_image_file, "PNG")
            output_image_file.flush()
            return self.image_to_text(output_image_file.name)

    """创建从图片中提取文本的函数
    """

    def image_to_text(self, image_path: str) -> str:
        # 从图片中抽取文本
        result = self.image_reader.readtext(image_path)
        predictions = "".join([item[1] for item in result])
        return predictions

    """从页面中提取表格内容
    """

    @staticmethod
    def extract_table(pdf: pdfplumber.PDF, page_num: int, table_num: int) -> List[Any]:
        # 查找已检查的页面
        table_page = pdf.pages[page_num]
        # 提取适当的表格
        table = table_page.extract_tables()[table_num]
        return table

    """合并分页表格
    """

    @staticmethod
    def merge_page_tables(total_tables: List[PageItem]) -> List[PageItem]:
        # 合并分页表格
        i = len(total_tables) - 1
        while i - 1 >= 0:
            table = total_tables[i]
            pre_table = total_tables[i - 1]
            if table["page_number"] == pre_table["page_number"]:
                continue
            if table["page_number"] - pre_table["page_number"] > 1:
                continue
            if (
                table["index_id"] <= 1
                and abs(table["element"].bbox[0] - pre_table["element"].bbox[0]) < 1
                and abs(table["element"].bbox[2] - pre_table["element"].bbox[2]) < 1
            ):
                total_tables[i - 1]["text"].extend(total_tables[i]["text"])
                del total_tables[i]
            i -= 1
        return total_tables

    """将表格转换为适当的格式
    """

    @staticmethod
    def parse_table(table: List[List]) -> str:
        table_string = ""
        # 遍历表格的每一行
        for row_num in range(len(table)):
            row = table[row_num]
            # 从warp的文字删除线路断路器
            cleaned_row = [
                item.replace("\n", " ")
                if item is not None and "\n" in item
                else "None"
                if item is None
                else item
                for item in row
            ]
            # 将表格转换为字符串，注意'|'、'\n'
            table_string += "|" + "|".join(cleaned_row) + "|" + "\n"
        # 删除最后一个换行符
        table_string = table_string.strip()
        return table_string

    """为表格生成摘要
    """

    @staticmethod
    def tables_summarize(table: List[List]) -> str:
        prompt_text = f"请为以下表格生成一个摘要: {table}"
        response = Settings.llm.complete(
            prompt_text,
            max_tokens=200,  # 调整为所需的摘要长度
            n=1,  # 生成摘要的数量
        )
        summarized_text = response
        return summarized_text

    """表格数据转化为json数据
    """

    @staticmethod
    def table_to_json(table: List[List]) -> str:
        # 提取表头
        table_info = []
        column_name = table[0]
        for row in range(1, len(table)):
            single_line_dict = {}
            for column in range(len(column_name)):
                if column_name[column] and len(column_name[column]) > 0:
                    single_line_dict[column_name[column]] = table[row][column]
            table_info.append(single_line_dict)

        return json.dumps(table_info, ensure_ascii=False)

    """创建一个文本提取函数
    """

    @staticmethod
    def text_extraction(elements: List[LTTextBoxHorizontal]) -> List[str]:
        # 找到每一行的坐标
        boxes, texts = [], []
        # 页面文字的开始和结束坐标
        max_x1 = 0
        min_x0 = sys.maxsize
        for text_box_h in elements:
            if isinstance(text_box_h, LTTextBoxHorizontal):
                for text_box_h_l in text_box_h:
                    if isinstance(text_box_h_l, LTTextLineHorizontal):
                        x0, y0, x1, y1 = text_box_h_l.bbox
                        text = text_box_h_l.get_text()
                        # 判断这一行是否以标点符号结尾。以标点符号结尾的行的结束位置和正常文字的结束位置不同
                        if not (
                            text[-1] == "\n"
                            and len(text) >= 2
                            and unicodedata.category(text[-2]).startswith("P")
                        ):
                            max_x1 = max(max_x1, x1)
                        min_x0 = min(min_x0, x0)
                        texts.append(text)
                        boxes.append((x0, x1))
        # 判断是否去除换行符的条件：该行的结尾坐标大于等于除标点符号结尾的行的坐标向下取整 且 下一行的开头坐标小于等于最小文字坐标取整+1
        for cur in range(len(boxes) - 1):
            if boxes[cur][1] >= int(max_x1) and boxes[cur + 1][0] <= int(min_x0) + 1:
                texts[cur] = texts[cur].replace("\n", "")

        return texts

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

        # check if file_path is a string or Path
        if not isinstance(file_path, str) and not isinstance(file_path, Path):
            raise TypeError("file_path must be a string or Path.")
        # open PDF file

        pdfFileObj = open(file_path, "rb")
        # 创建一个PDF阅读器对象
        pdf_read = PyPDF2.PdfReader(pdfFileObj)

        total_tables = []
        page_items = []
        # 打开pdf文件
        pdf = pdfplumber.open(file_path)
        # 从PDF中提取页面
        for pagenum, page in enumerate(extract_pages(file_path)):
            # 初始化从页面中提取文本所需的变量
            page_object = pdf_read.pages[pagenum]
            text_elements = []
            text_from_images = []
            # 初始化检查表的数量
            table_num = 0
            first_element = True
            # 查找已检查的页面
            page_tables = pdf.pages[pagenum]
            # 找出本页上的表格数目
            tables = page_tables.find_tables()

            # 找到所有的元素
            page_elements = [(element.y1, element) for element in page._objs]
            # 对页面中出现的所有元素进行排序
            page_elements.sort(key=lambda a: a[0], reverse=True)

            # 查找组成页面的元素
            for i, component in enumerate(page_elements):
                # 提取页面布局的元素
                element = component[1]

                # 检查该元素是否为文本元素
                if isinstance(element, LTTextBoxHorizontal):
                    text_elements.append(element)

                # 检查元素中的图像
                elif isinstance(element, LTFigure) and self.enable_image_ocr:
                    # 从PDF中提取文字
                    image_texts = self.process_image(element, page_object)
                    text_from_images.append(image_texts)

                # 检查表的元素
                elif isinstance(element, LTRect):
                    lower_side = sys.maxsize
                    upper_side = 0

                    # 如果第一个矩形元素
                    if first_element is True and (table_num + 1) <= len(tables):
                        # 找到表格的边界框
                        lower_side = page.bbox[3] - tables[table_num].bbox[3]
                        upper_side = element.y1
                        # 从表中提取信息
                        tabel_text = PaiPDFReader.extract_table(pdf, pagenum, table_num)

                        item = PageItem(
                            page_number=pagenum,
                            index_id=i,
                            item_type="table",
                            element=element,
                            table_num=table_num,
                            text=tabel_text,
                        )
                        total_tables.append(item)
                        # 让它成为另一个元素
                        first_element = False

                    # 检查我们是否已经从页面中提取了表
                    if element.y0 >= lower_side and element.y1 <= upper_side:
                        pass
                    elif not isinstance(page_elements[i + 1][1], LTRect):
                        first_element = True
                        table_num += 1

            # 文本处理
            text_from_texts = PaiPDFReader.text_extraction(text_elements)
            page_plain_text = "".join(text_from_texts)
            # 图片处理
            page_image_text = "".join(text_from_images)

            page_items.append(
                (
                    PageItem(
                        page_number=pagenum, item_type="text", text=page_plain_text
                    ),
                    PageItem(
                        page_number=pagenum,
                        item_type="image_text",
                        text=page_image_text,
                    ),
                )
            )

        # 合并分页表格
        total_tables = PaiPDFReader.merge_page_tables(total_tables)

        # 构造返回数据
        docs = []
        for pagenum, item in enumerate(page_items):
            page_tables_texts = []
            page_tables_summaries = []
            page_tables_json = []
            for table in total_tables:
                # 如果页面匹配
                if pagenum == table["page_number"]:
                    # 将表信息转换为结构化字符串格式
                    table_string = PaiPDFReader.parse_table(table["text"])
                    summarized_table_text = PaiPDFReader.tables_summarize(table["text"])
                    json_data = PaiPDFReader.table_to_json(table["text"])
                    page_tables_texts.append(table_string)
                    page_tables_summaries.append(summarized_table_text.text)
                    page_tables_json.append(json_data)
            page_table_text = "".join(page_tables_texts)
            page_table_summary = "".join(page_tables_summaries)
            page_table_json = "".join(page_tables_json)

            page_info_text = item[0]["text"] + item[1]["text"] + page_table_text

            # if extra_info is not None, check if it is a dictionary
            if extra_info:
                if not isinstance(extra_info, dict):
                    raise TypeError("extra_info must be a dictionary.")

            if metadata:
                if not extra_info:
                    extra_info = {}
                extra_info["total_pages"] = len(pdf_read.pages)
                extra_info["file_path"] = str(file_path)
                extra_info["table_summary"] = page_table_summary
                extra_info["table_json"] = page_table_json

                doc = Document(
                    text=page_info_text,
                    extra_info=dict(
                        extra_info,
                        **{
                            "source": f"{pagenum + 1}",
                        },
                    ),
                )

                doc.excluded_embed_metadata_keys.append("table_json")
                doc.excluded_llm_metadata_keys.append("table_json")

                docs.append(doc)
            else:
                doc = Document(
                    text=page_info_text,
                    extra_info=dict(
                        extra_info,
                        **{
                            "source": f"{pagenum + 1}",
                        },
                    ),
                )
                docs.append(doc)
        # return list of documents
        return docs
