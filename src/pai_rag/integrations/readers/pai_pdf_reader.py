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
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
import json
import unicodedata
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

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
        self, enable_image_ocr: bool = False, model_dir: str = DEFAULT_MODEL_DIR
    ) -> None:
        self.enable_image_ocr = enable_image_ocr
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
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".pdf"
        ) as cropped_pdf_file:
            cropped_pdf_writer.add_page(page_object)
            cropped_pdf_writer.write(cropped_pdf_file)
            cropped_pdf_file.flush()
            # Return the OCR-processed text
            return self.ocr_pdf(cropped_pdf_file.name)

    def ocr_pdf(self, input_file: str) -> str:
        """
        Function to convert PDF content into an image and then perform OCR (Optical Character Recognition)

        Args:
            input_file (str): input file path.

        Returns:
             str: text from ocr.
        """
        images = convert_from_path(input_file)
        image = images[0]
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".png"
        ) as output_image_file:
            image.save(output_image_file, "PNG")
            output_image_file.flush()
            return self.image_to_text(output_image_file.name)

    def image_to_text(self, image_path: str) -> str:
        """
        Function to perform OCR to extract text from image

        Args:
            image_path (str): input image path.

        Returns:
            str: text from ocr.
        """
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

    """Function to parse table
    """

    @staticmethod
    def parse_table(table: List[List]) -> str:
        table_string = ""
        for row_num in range(len(table)):
            row = table[row_num]
            cleaned_row = [
                item.replace("\n", " ")
                if item is not None and "\n" in item
                else "None"
                if item is None
                else item
                for item in row
            ]
            table_string += "|" + "|".join(cleaned_row) + "|" + "\n"
        table_string = table_string.strip()
        return table_string

    """Function to summarize table
    """

    @staticmethod
    def tables_summarize(table: List[List]) -> str:
        prompt_text = f"请为以下表格生成一个摘要: {table}"
        response = Settings.llm.complete(
            prompt_text,
            max_tokens=200,
            n=1,
        )
        summarized_text = response
        return summarized_text

    """Function to convert table data to json
    """

    @staticmethod
    def table_to_json(table: List[List]) -> str:
        table_info = []
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
        for pagenum, page in enumerate(extract_pages(file_path)):
            # Initialize variables for extracting text from the page
            page_object = pdf_read.pages[pagenum]
            text_elements = []
            text_from_images = []
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

            # Iterate through the page's elements
            for i, component in enumerate(page_elements):
                # Extract text elements
                element = component[1]

                # Check if the element is a text box
                if isinstance(element, LTTextBoxHorizontal):
                    text_elements.append(element)

                # Check for images and extract text from them if OCR is enabled
                elif isinstance(element, LTFigure) and self.enable_image_ocr:
                    # Extract text from the PDF image
                    image_texts = self.process_pdf_image(element, page_object)
                    text_from_images.append(image_texts)

                # Check for table elements
                elif isinstance(element, LTRect):
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
            # Image text extraction
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

        # Merge tables across pages
        total_tables = PaiPDFReader.merge_page_tables(total_tables)

        # Construct the returned data
        docs = []
        for pagenum, item in enumerate(page_items):
            page_tables_texts = []
            page_tables_summaries = []
            page_tables_json = []
            for table in total_tables:
                # If the page number matches
                if pagenum == table["page_number"]:
                    # Convert the table data to a structured string
                    table_string = PaiPDFReader.parse_table(table["text"])
                    summarized_table_text = PaiPDFReader.tables_summarize(table["text"])
                    json_data = PaiPDFReader.table_to_json(table["text"])
                    page_tables_texts.append(table_string)
                    page_tables_summaries.append(
                        summarized_table_text.text[:TABLE_SUMMARY_MAX_TOKEN]
                    )
                    page_tables_json.append(json_data)
            page_table_text = "".join(page_tables_texts)
            page_table_summary = "".join(page_tables_summaries)
            page_table_json = "".join(page_tables_json)

            page_info_text = item[0]["text"] + item[1]["text"] + page_table_text

            # if `extra_info` is not None, check if it is a dictionary
            if extra_info:
                if not isinstance(extra_info, dict):
                    raise TypeError("extra_info must be a dictionary.")

            if metadata:
                if not extra_info:
                    extra_info = {}
                extra_info["total_pages"] = len(pdf_read.pages)
                extra_info["file_path"] = str(file_path)
                extra_info["table_summary"] = page_table_summary[
                    :PAGE_TABLE_SUMMARY_MAX_TOKEN
                ]
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
