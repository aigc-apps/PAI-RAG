import hashlib
from io import BytesIO
import re
from docx import Document as DocxDocument
from loguru import logger
from rag.file.readers.base import BaseReader, FileItem, Document, List
from rag.file.store.base import BaseFileStore
from rag.file.store.oss_store import OssFileStore
from rag.file.utils.image_utils import compress_image_if_needed
from rag.file.utils.markdown_utils import (
    PaiTable,
    convert_table_to_markdown,
    is_horizontal_table,
)
from rag.file.image_caption_tool import ImageCaptionTool


class DocxReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, image_caption_tool: ImageCaptionTool = None
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        logger.info("DocxReader inited.")

    def _convert_paragraph(self, paragraph):
        text = paragraph.text.strip()
        if not text:
            return ""

        # 处理标题
        if paragraph.style and paragraph.style.name.startswith("Heading"):
            heading_level = int(
                re.search(r"Heading (\d)", paragraph.style.name).group(1)
            )
            if heading_level > 6:
                heading_level = 6
            return f"{'#' * heading_level} {text}\n\n"

        # 处理普通段落
        return f"{text}\n\n"

    def _get_list_level(self, paragraph):
        indent_levels = {
            "List Paragraph": 0,
            "List Bullet": 1,
            "List Number": 1,
            "List Bullet 2": 2,
            "List Number 2": 2,
            "List Bullet 3": 3,
            "List Number 3": 3,
        }

        # 获取段落的样式名称
        style_name = paragraph.style.name
        # 根据样式名称获取层级
        return indent_levels.get(style_name, 0)

    def _convert_list(self, paragraph, level=0):
        text = paragraph.text.strip()
        if not text:
            return ""

        # 处理无序列表
        if paragraph.style.name.startswith("List Bullet"):
            return f"{'-' * level} {text}\n"

        # 处理有序列表
        if paragraph.style.name.startswith("List Number"):
            return f"{level}. {text}\n"

        return ""

    def _convert_table_to_markdown(self, table, doc_name):
        total_cols = max(len(row.cells) for row in table.rows)

        table_matrix = []
        for row in table.rows:
            table_matrix.append(self._parse_row(row, doc_name, total_cols))
        if is_horizontal_table(table_matrix):
            table = PaiTable(data=table_matrix, row_headers_index=[0])
        else:
            table = PaiTable(data=table_matrix, column_headers_index=[0])
        return convert_table_to_markdown(table, total_cols)

    def _parse_row(self, row, doc_name, total_cols):
        row_cells = [""] * total_cols
        col_index = 0
        for cell in row.cells:
            while col_index < total_cols and row_cells[col_index] != "":
                col_index += 1
            if col_index >= total_cols:
                break
            cell_content = self._parse_cell(cell, doc_name).strip()
            row_cells[col_index] = cell_content
            col_index += 1
        return row_cells

    def _parse_cell(self, cell, doc_name):
        cell_content = []
        for paragraph in cell.paragraphs:
            parsed_paragraph = self._parse_cell_paragraph(paragraph, doc_name)
            if parsed_paragraph:
                cell_content.append(parsed_paragraph)
        unique_content = list(dict.fromkeys(cell_content))
        return " ".join(unique_content)

    def _parse_cell_paragraph(self, paragraph, doc_name):
        paragraph_content = []
        for run in paragraph.runs:
            if not run.element.xpath(".//a:blip"):
                paragraph_content.append(run.text)
        return "".join(paragraph_content).strip()

    def convert_docx_to_markdown(
        self, document: DocxDocument, save_name_template: str
    ) -> str:
        paragraphs = document.paragraphs.copy()
        tables = document.tables.copy()
        markdown = []
        images = []
        for element in document.element.body:
            if isinstance(element.tag, str) and element.tag.endswith("p"):  # 段落
                paragraph = paragraphs.pop(0)

                if paragraph.style and paragraph.style.name.startswith("List"):
                    current_list_level = self._get_list_level(paragraph)
                    markdown.append(self._convert_list(paragraph, current_list_level))
                else:
                    for run in paragraph.runs:
                        if (
                            hasattr(run.element, "tag")
                            and isinstance(element.tag, str)
                            and run.element.tag.endswith("r")
                        ):
                            drawing_elements = run.element.findall(
                                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"
                            )
                            for drawing in drawing_elements:
                                blip_elements = drawing.findall(
                                    ".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
                                )
                                for blip in blip_elements:
                                    embed_id = blip.get(
                                        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                                    )
                                    if (
                                        embed_id
                                        and isinstance(self.file_store, OssFileStore)
                                        and self.image_caption_tool
                                    ):
                                        image_part = document.part.related_parts.get(
                                            embed_id
                                        )
                                        if hasattr(image_part, "blob"):
                                            image_blob = image_part.blob
                                            image_name = (
                                                hashlib.md5(image_blob).hexdigest()
                                                + ".jpeg"
                                            )

                                            save_image_name = save_name_template.format(
                                                image_name
                                            )
                                            image_file = BytesIO(image_blob)
                                            image_file = compress_image_if_needed(
                                                image_file
                                            )
                                            if not image_file:
                                                continue
                                            try:
                                                self.file_store.save(
                                                    image_file, save_image_name
                                                )
                                                image_alt_text = (
                                                    self.image_caption_tool.extract_url(
                                                        self.file_store.get_url(
                                                            save_image_name
                                                        )
                                                    )
                                                )
                                                image_text = f'<img src="{save_image_name}" alt="{image_alt_text}">'
                                                markdown.append(f"{image_text}\n\n")
                                                images.append(save_image_name)

                                                logger.info(
                                                    f"Successfully saved image {save_image_name}."
                                                )
                                            except Exception as ex:
                                                logger.exception(
                                                    f"Failed to save image from URL: {save_image_name}. Error: {ex}"
                                                )

                    markdown.append(self._convert_paragraph(paragraph))

            elif isinstance(element.tag, str) and element.tag.endswith("tbl"):  # 表格
                table = tables.pop(0)
                markdown.append(self._convert_table_to_markdown(table, None))
                markdown.append("\n\n")

        return "".join(markdown), images

    def read(self, file_item: FileItem) -> List[Document]:
        """
        Read a CSV file and return a list of Documents.
        """
        try:
            file_item.file.seek(0)
            docx_file = DocxDocument(file_item.file)

            markdown_content, images = self.convert_docx_to_markdown(
                docx_file, file_item.kb_id + "/images/{}"
            )

            metadata = file_item.metadata()
            metadata["images"] = images

            docs = [Document(id_=file_item.id, text=markdown_content, metadata=metadata)]
            logger.info(f"Successfully read {file_item.file_name}.")

            return docs
        except Exception as e:
            logger.exception(e)
            return []
