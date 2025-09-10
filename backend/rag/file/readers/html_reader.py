import re
from bs4 import BeautifulSoup
import html2text
from loguru import logger
from rag.file.readers.base import BaseReader, FileItem, Document, List
from rag.file.store.base import BaseFileStore
from rag.file.store.oss_store import OssFileStore
from rag.file.utils.image_utils import get_image_from_url
from rag.file.utils.markdown_utils import PaiTable, convert_table_to_markdown
from rag.file.image_caption_tool import ImageCaptionTool

MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\((https?://[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))\)",
    re.IGNORECASE,
)


class HtmlReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, image_caption_tool: ImageCaptionTool = None
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        logger.info("HtmlReader inited.")

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

    def _replace_image_paths(self, content: str, save_name_template: str):
        image_matches = MARKDOWN_IMAGE_PATTERN.finditer(content)
        saved_images = []
        for match in image_matches:
            full_match = match.group(0)  # 整个匹配
            image_url = match.group(1)  # 捕获的URL
            image_file, image_name = get_image_from_url(image_url)
            if image_name:
                save_image_name = save_name_template.format(image_name)

                try:
                    self.file_store.save(image_file, save_image_name)
                    image_alt_text = self.image_caption_tool.extract_url(
                        self.file_store.get_url(save_image_name)
                    )
                    cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
                    image_text = f'\n![{cleaned_alt}]({save_image_name})\n'
                    content = content.replace(
                        full_match,
                        image_text,
                    )
                    saved_images.append(save_image_name)

                    logger.info(
                        f"Successfully saved image {save_image_name} from URL: {image_url}"
                    )
                except Exception as ex:
                    logger.exception(
                        f"Failed to save image from URL: {image_url}. Error: {ex}"
                    )

        return content, saved_images

    def read(self, file_item: FileItem) -> List[Document]:
        """
        Read a CSV file and return a list of Documents.
        """
        try:
            file_item.file.seek(0)
            html_content = file_item.file.read().decode("utf-8")

            modified_html, tables = self._extract_tables(html_content)
            h = html2text.HTML2Text()

            # 配置 html2text 对象
            h.ignore_links = True  # 是否忽略链接
            h.ignore_images = False  # 是否忽略图片
            # h.escape_all = True  # 是否转义所有特殊字符
            h.body_width = 0  # 设置行宽为 0 表示不限制行宽

            # 将 HTML 转换为 Markdown
            markdown_content = h.handle(modified_html)
            for table in tables:
                table_markdown = self._convert_table_to_markdown(table) + "\n\n"
                placeholder = f"<!-- TABLE_PLACEHOLDER_{id(table)} -->"
                markdown_content = markdown_content.replace(placeholder, table_markdown)

            images = []
            if isinstance(self.file_store, OssFileStore) and self.image_caption_tool:
                markdown_content, images = self._replace_image_paths(
                    markdown_content, file_item.kb_id + "/images/{}"
                )
                logger.info(
                    f"Successfully read {file_item.file_name} with images {images}."
                )

            metadata = file_item.metadata()

            docs = [Document(id_=file_item.id, text=markdown_content, metadata=metadata)]
            logger.info(f"Successfully read {file_item.file_name}.")

            return docs
        except Exception as e:
            logger.exception(e)
            return []
