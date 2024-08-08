"""Read PDF files."""

from pathlib import Path
from typing import Dict, List, Optional, Union, TypedDict, Any
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument
import PyPDF2
from PyPDF2 import PageObject
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTRect,
    LTFigure,
    LTTextBoxHorizontal,
    LTTextLineHorizontal,
    LTLine,
)
import pdfplumber
from pdf2image import convert_from_bytes
import easyocr
from llama_index.core import Settings
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
import json
import unicodedata
import logging
import cv2
import os
from tqdm import tqdm
from io import BytesIO

logger = logging.getLogger(__name__)

TABLE_SUMMARY_MAX_ROW_NUM = 5
TABLE_SUMMARY_MAX_COL_NUM = 10
TABLE_SUMMARY_MAX_CELL_TOKEN = 20
TABLE_SUMMARY_MAX_TOKEN = 200
PAGE_TABLE_SUMMARY_MAX_TOKEN = 400


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
        self,
        enable_image_ocr: bool = False,
        enable_table_summary: bool = False,
        model_dir: str = DEFAULT_MODEL_DIR,
        oss_cache: Any = None,
    ) -> None:
        self.enable_image_ocr = enable_image_ocr
        self.enable_table_summary = enable_table_summary
        self._oss_cache = oss_cache
        if self.enable_table_summary:
            logger.info("process with table summary")
        if self.enable_image_ocr:
            self.model_dir = model_dir or os.path.join(DEFAULT_MODEL_DIR, "easyocr")
            logger.info("start loading ocr model")
            self.image_reader = easyocr.Reader(
                ["ch_sim", "en"],
                model_storage_directory=self.model_dir,
                download_enabled=True,
                detector=True,
                recognizer=True,
            )
            logger.info("finished loading ocr model")

    def process_pdf_image(self, element: LTFigure, page_object: PageObject) -> str:
        """
        Processes an image element from a PDF, crops it out, and performs OCR on the result.

        Args:
            element (LTFigure): An LTFigure object representing the image in the PDF, containing its coordinates.
            page_object (PageObject): A PageObject representing the page in the PDF to be cropped.

        Returns:
            str: The OCR-processed text from the cropped image.
        """
        assert (
            self._oss_cache is not None
        ), "Oss config must be provided for image processing."

        # Retrieve the image's coordinates
        [image_left, image_top, image_right, image_bottom] = [
            element.x0,
            element.y0,
            element.x1,
            element.y1,
        ]
        # Adjust the page's media box to crop the image based on the coordinates
        page_object.mediabox.lower_left = (image_left, image_bottom)
        page_object.mediabox.upper_right = (image_right, image_top)
        # Save the cropped page as a new PDF file and perform OCR
        cropped_pdf_writer = PyPDF2.PdfWriter()
        cropped_pdf_stream = BytesIO()

        cropped_pdf_writer.add_page(page_object)
        cropped_pdf_writer.write(cropped_pdf_stream)
        cropped_pdf_stream.seek(0)

        image = convert_from_bytes(cropped_pdf_stream.getvalue())[0]

        # Check image size
        if image.width <= 200 or image.height <= 200:
            logger.info(
                f"Skip crop image with width={image.width}, height={image.height}."
            )
            return None

        image_stream = BytesIO()
        image.save(image_stream, format="jpeg")

        image_stream.seek(0)
        data = image_stream.getvalue()

        image_url = self._oss_cache.put_object_if_not_exists(
            data=data,
            file_ext=".jpeg",
            headers={
                "x-oss-object-acl": "public-read"
            },  # set public read to make image accessible
            path_prefix="pairag/pdf_images/",
        )

        logger.info(
            f"Cropped image {image_url} with width={image.width}, height={image.height}."
        )
        return image_url

    def image_to_text(self, image_path: str) -> str:
        """
        Function to perform OCR to extract text from image

        Args:
            image_path (str): input image path.

        Returns:
            str: text from ocr.
        """
        image = cv2.imread(image_path)
        if image is None or image.shape[0] <= 1 or image.shape[1] <= 1:
            return ""

        result = self.image_reader.readtext(image_path)
        predictions = "".join([item[1] for item in result])
        return predictions

    """Function to extract content from table
    """

    @staticmethod
    def extract_table(pdf: pdfplumber.PDF, page_num: int, table_num: int) -> List[Any]:
        table_page = pdf.pages[page_num]
        table = table_page.extract_tables()[table_num]
        return table

    """Function to merge paginated tables
    """

    @staticmethod
    def merge_page_tables(total_tables: List[PageItem]) -> List[PageItem]:
        i = len(total_tables) - 1
        while i - 1 >= 0:
            table = total_tables[i]
            pre_table = total_tables[i - 1]
            if table["page_number"] == pre_table["page_number"]:
                i -= 1
                continue
            if table["page_number"] - pre_table["page_number"] > 1:
                i -= 1
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

    """Function to summarize table
    """

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
        if not PaiPDFReader.is_horizontal_table(table):
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
        return summarized_text

    @staticmethod
    def is_horizontal_table(table: List[List]) -> bool:
        # if the table is empty or the first (header) of table is empty, it's not a horizontal table
        if not table or not table[0]:
            return False

        vertical_value_any_count = 0
        horizontal_value_any_count = 0
        vertical_value_all_count = 0
        horizontal_value_all_count = 0

        """If it is a horizontal table, the probability that each row contains at least one number is higher than the probability that each column contains at least one number.
        If it is a horizontal table with headers, the number of rows that are entirely composed of numbers will be greater than the number of columns that are entirely composed of numbers.
        """

        for row in table:
            if any(isinstance(item, (int, float)) for item in row):
                horizontal_value_any_count += 1
            if all(isinstance(item, (int, float)) for item in row):
                horizontal_value_all_count += 1

        for col in zip(*table):
            if any(isinstance(item, (int, float)) for item in col):
                vertical_value_any_count += 1
            if all(isinstance(item, (int, float)) for item in col):
                vertical_value_all_count += 1

        return (
            horizontal_value_any_count >= vertical_value_any_count
            or horizontal_value_all_count > 0 >= vertical_value_all_count
        )

    """Function to convert table data to json
       """

    @staticmethod
    def table_to_json(table: List[List]) -> str:
        table_info = []
        if not PaiPDFReader.is_horizontal_table(table):
            table = list(zip(*table))
        column_name = table[0]
        for row in range(1, len(table)):
            single_line_dict = {}
            for column in range(len(column_name)):
                if (
                    column_name[column]
                    and len(column_name[column]) > 0
                    and column < len(table[row])
                ):
                    single_line_dict[column_name[column]] = table[row][column]
            table_info.append(single_line_dict)

        return json.dumps(table_info, ensure_ascii=False)

    """Function to process text in pdf
    """

    @staticmethod
    def text_extraction(elements: List[LTTextBoxHorizontal]) -> List[str]:
        """
        Extracts text lines from a list of text boxes and handles line breaks under specific conditions.

        Args:
            elements: A list of LTTextBoxHorizontal objects representing text boxes on a page.

        Returns:
            A list containing the extracted text lines with line breaks removed as per defined conditions.
        """
        boxes, texts = [], []
        # Initialize the start and end coordinates of the page text
        max_x1 = 0
        min_x0 = float("inf")
        for text_box_h in elements:
            if isinstance(text_box_h, LTTextBoxHorizontal):
                for text_box_h_l in text_box_h:
                    if isinstance(text_box_h_l, LTTextLineHorizontal):
                        # Process each text line's coordinates and content
                        x0, y0, x1, y1 = text_box_h_l.bbox
                        text = text_box_h_l.get_text()
                        # Check if the line ends with punctuation and requires special handling
                        if not (
                            text[-1] == "\n"
                            and len(text) >= 2
                            and unicodedata.category(text[-2]).startswith("P")
                        ):
                            max_x1 = max(max_x1, x1)
                        min_x0 = min(min_x0, x0)
                        texts.append(text)
                        boxes.append((x0, x1))
        # Remove line breaks based on defined conditions
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
        # Create a PDF reader object
        pdf_read = PyPDF2.PdfReader(pdfFileObj)

        total_tables = []
        page_items = []
        # Open the PDF and extract pages
        pdf = pdfplumber.open(file_path)
        pages = tqdm(
            extract_pages(file_path), desc="processing pages in pdf", unit="page"
        )
        image_documents = []

        file_name = os.path.basename(file_path)
        for pagenum, page in enumerate(pages):
            extra_info["file_path"] = f"_page_{pagenum + 1}".join(
                os.path.splitext(file_path)
            )

            # Initialize variables for extracting text from the page
            page_object = pdf_read.pages[pagenum]
            text_elements = []
            # Initialize table count
            table_num = 0
            first_element = True
            # Find the checked page
            page_tables = pdf.pages[pagenum]
            # Find the number of tables on the page
            tables = page_tables.find_tables()

            # Find all elements on the page
            page_elements = [(element.y1, element) for element in page._objs]
            # Sort the elements on the page by their y1 coordinate
            page_elements.sort(key=lambda a: a[0], reverse=True)

            image_cnt = 0
            # Iterate through the page's elements
            for i, component in enumerate(page_elements):
                # Extract text elements
                element = component[1]

                # Check if the element is a text box
                if isinstance(element, LTTextBoxHorizontal):
                    text_elements.append(element)

                elif isinstance(element, LTFigure):
                    image_url = self.process_pdf_image(element, page_object)
                    if image_url:
                        image_cnt += 1
                        extra_info[
                            "file_name"
                        ] = f"_page_{pagenum + 1}_image_{image_cnt}".join(
                            os.path.splitext(file_name)
                        )
                        image_documents.append(
                            ImageDocument(
                                image_url=image_url,
                                image_mimetype="image/jpeg",
                                extra_info={"image_url": image_url, **extra_info},
                            )
                        )

                # Check for table elements
                elif isinstance(element, LTRect) or isinstance(element, LTLine):
                    lower_side = float("inf")
                    upper_side = 0

                    # If it's the first rectangle element
                    if first_element is True and (table_num + 1) <= len(tables):
                        # Find the bounding box of the table
                        lower_side = page.bbox[3] - tables[table_num].bbox[3]
                        upper_side = element.y1
                        # Extract the table data
                        table_text = PaiPDFReader.extract_table(pdf, pagenum, table_num)

                        item = PageItem(
                            page_number=pagenum,
                            index_id=i,
                            item_type="table",
                            element=element,
                            table_num=table_num,
                            text=table_text,
                        )
                        total_tables.append(item)
                        # Move to the next element
                        first_element = False

                    # Check if we've extracted a table from the page
                    if element.y0 >= lower_side and element.y1 <= upper_side:
                        pass
                    elif i + 1 < len(page_elements) and not isinstance(
                        page_elements[i + 1][1], LTRect
                    ):
                        first_element = True
                        table_num += 1

            # Text extraction from text elements
            text_from_texts = PaiPDFReader.text_extraction(text_elements)
            page_plain_text = "".join(text_from_texts)

            page_items.append(
                PageItem(page_number=pagenum, item_type="text", text=page_plain_text)
            )

        # Merge tables across pages
        total_tables = PaiPDFReader.merge_page_tables(total_tables)

        # Construct the returned data
        docs = []
        page_items = tqdm(page_items, desc="processing tables in pages", unit="page")
        for pagenum, item in enumerate(page_items):
            page_tables_summaries = []
            page_tables_json = []
            for table in total_tables:
                # If the page number matches
                if pagenum == table["page_number"]:
                    if self.enable_table_summary:
                        summarized_table_text = PaiPDFReader.tables_summarize(
                            table["text"]
                        )
                        page_tables_summaries.append(
                            summarized_table_text.text[:TABLE_SUMMARY_MAX_TOKEN]
                        )
                    json_data = PaiPDFReader.table_to_json(table["text"])
                    page_tables_json.append(json_data)
            page_table_summary = "\n".join(page_tables_summaries)
            page_table_json = "\n".join(page_tables_json)

            page_info_text = item["text"] + page_table_json
            if not page_info_text:
                # Skip empty page
                continue

            # if `extra_info` is not None, check if it is a dictionary
            if extra_info:
                if not isinstance(extra_info, dict):
                    raise TypeError("extra_info must be a dictionary.")

            if metadata:
                if not extra_info:
                    extra_info = {}
                extra_info["total_pages"] = len(pdf_read.pages)
                table_summary = page_table_summary[:PAGE_TABLE_SUMMARY_MAX_TOKEN]
                if table_summary:
                    extra_info["table_summary"] = table_summary
                extra_info["file_name"] = f"_page_{pagenum + 1}".join(
                    os.path.splitext(file_name)
                )

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

        docs.extend(image_documents)
        return docs
