import hashlib
from io import BytesIO
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from loguru import logger
from rag.file.readers.base import BaseReader, FileItem, Document, List
from rag.file.store.base import BaseFileStore
from rag.file.store.oss_store import OssFileStore
from rag.file.utils.image_utils import compress_image_if_needed
from rag.file.utils.markdown_utils import PaiTable, convert_table_to_markdown
from rag.file.image_caption_tool import ImageCaptionTool


class PptxReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, image_caption_tool: ImageCaptionTool = None
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        logger.info("PptxReader inited.")

    def _extract_shape(self, slide_number, shape, save_name_template: str):
        markdown = []
        images = []
        if shape.name.startswith("Title"):
            # 标题
            markdown.append(f"# {shape.text}\n\n")
        elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            # 图片
            logger.info("extracting image from pptx.")
            if isinstance(self.file_store, OssFileStore) and self.image_caption_tool:
                image_blob = shape.image.blob

                image_name = hashlib.md5(image_blob).hexdigest() + ".jpeg"
                save_image_name = save_name_template.format(image_name)
                image_file = BytesIO(image_blob)
                image_file = compress_image_if_needed(image_file)
                if image_file:
                    try:
                        self.file_store.save(BytesIO(image_blob), save_image_name)
                        image_alt_text = self.image_caption_tool.extract_url(
                            self.file_store.get_url(save_image_name)
                        )
                        image_text = (
                            f'<img src="{save_image_name}" alt="{image_alt_text}">'
                        )
                        markdown.append(f"{image_text}\n\n")
                        images.append(save_image_name)

                        logger.info(f"Successfully saved image {save_image_name}.")
                    except Exception as ex:
                        logger.exception(
                            f"Failed to save image from URL: {save_image_name}. Error: {ex}"
                        )
        elif shape.shape_type == MSO_SHAPE_TYPE.TEXT_BOX:
            # 文本框
            markdown.append(f"{shape.text}\n\n")
        elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            # 表格
            table = shape.table
            markdown.append(self._convert_table_to_pai_table(table))
            markdown.append("\n\n")
        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            texts = []
            for p in sorted(shape.shapes, key=lambda x: (x.top // 10, x.left)):
                md, new_images = self._extract_shape(
                    slide_number, p, save_name_template
                )
                if md:
                    texts.append(md)
                    images.extend(new_images)

            markdown.append("\n".join(texts))

        return "".join(markdown), images

    def _convert_table_to_pai_table(self, table):
        table_matrix = [
            ["" for _ in range(len(table.columns))] for _ in range(len(table.rows))
        ]
        visited_cells = set()
        for i in range(len(table.rows)):
            for j in range(len(table.columns)):
                if (i, j) in visited_cells:
                    continue
                cell_content = table.cell(i, j).text.replace("\n", "").replace("\r", "")
                if table.cell(i, j).is_merge_origin:
                    col_span = table.cell(i, j).span_width
                    row_span = table.cell(i, j).span_height
                    while (
                        col_span > 1
                        and j + col_span <= len(table.columns)
                        and table_matrix[i][j + col_span - 1] == ""
                    ):
                        col_span -= 1
                        table_matrix[i][j + col_span] = cell_content
                        visited_cells.add((i, j + col_span))
                    while (
                        row_span > 1
                        and i + row_span <= len(table.rows)
                        and table_matrix[i + row_span - 1][j] == ""
                    ):
                        row_span -= 1
                        table_matrix[i + row_span][j] = cell_content
                        visited_cells.add((i + row_span, j))
                if table_matrix[i][j] == "":
                    table_matrix[i][j] = cell_content
                    visited_cells.add((i, j))

        row_headers_index = []
        col_headers_index = []
        if table.first_row:
            row_headers_index.append(0)
        if table.first_col:
            col_headers_index.append(0)
        pai_table = PaiTable(
            data=table_matrix,
            row_headers_index=row_headers_index,
            column_headers_index=col_headers_index,
        )
        return convert_table_to_markdown(pai_table, len(table.columns))

    def convert_pptx_to_markdown(
        self, presentation: Presentation, save_name_template: str
    ):
        markdown = []
        images = []
        slide_image_flag = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            image_flag = False
            for shape in slide.shapes:
                shape_markdown, shape_images = self._extract_shape(
                    slide_number, shape, save_name_template
                )
                markdown.append(shape_markdown)
                images.extend(shape_images)
            markdown.append(f"# slide_number_{slide_number}\n\n")
            slide_image_flag.append(image_flag)

        return "".join(markdown), images

    def read(self, file_item: FileItem) -> List[Document]:
        """
        Read a CSV file and return a list of Documents.
        """
        try:
            file_item.file.seek(0)
            presentation = Presentation(file_item.file)

            markdown_content, images = self.convert_pptx_to_markdown(
                presentation, file_item.kb_id + "/images/{}"
            )

            metadata = file_item.metadata()
            metadata["images"] = images

            docs = [Document(id_=file_item.id, text=markdown_content, metadata=metadata)]
            logger.info(f"Successfully read {file_item.file_name}.")

            return docs
        except Exception as e:
            logger.exception(e)
            return []
