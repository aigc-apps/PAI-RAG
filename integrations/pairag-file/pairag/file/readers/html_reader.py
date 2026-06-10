import re
from bs4 import BeautifulSoup
from markdownify import markdownify
from loguru import logger
from pairag.file.readers.base import BaseReader, FileItem, Document, List
from pairag.file.store.base import BaseFileStore
from pairag.file.utils.image_utils import get_image_from_url
from pairag.file.utils.markdown_tree_utils import PaiTable, convert_table_to_markdown
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.image_utils import to_markdown_image_text
from pairag.file.utils.text_utils import replace_consecutive_spaces

MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\((https?://[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff)(?:\?[^\s)]*)?)\)",
    re.IGNORECASE,
)

# Base64 image pattern: matches data:image/[type];base64,[data]
BASE64_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\((data:image/(?:png|jpe?g|gif|bmp|svg|webp|tiff);base64,[^\s)]+)\)",
    re.IGNORECASE,
)

# Limits for base64 image processing to prevent resource exhaustion
MAX_BASE64_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB max per image
MAX_BASE64_IMAGE_COUNT = 50  # Max 50 base64 images per document


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
        for idx, table in enumerate(tables):
            # 使用连字符而非下划线，避免markdownify自动转义
            placeholder = f"<!-- TABLE-PLACEHOLDER-{idx} -->"
            table.replace_with(placeholder)
            # Store the placeholder in the table element for later reference
            table.attrs['data-placeholder'] = placeholder
        return str(soup), tables

    def _pad_rows_to_max_cols(self, table_matrix, max_cols):
        for r in range(len(table_matrix)):
            if len(table_matrix[r]) < max_cols:
                table_matrix[r].extend([""] * (max_cols - len(table_matrix[r])))

    def _convert_table_to_pai_table(self, table):
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
            else:
                self._pad_rows_to_max_cols(table_matrix, max_cols)
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
                        and current_col_index < len(table_matrix[current_row_index])
                        and table_matrix[current_row_index][current_col_index] != ""
                    ):
                        current_col_index += 1
                if current_col_index + col_span > max_cols:
                    max_cols = current_col_index + col_span
                    self._pad_rows_to_max_cols(table_matrix, max_cols)
                for i in range(col_span):
                    if current_col_index + i < len(table_matrix[current_row_index]):
                        table_matrix[current_row_index][
                            current_col_index + i
                        ] = cell_content

                for i in range(1, row_span):
                    if current_row_index + i >= max_rows:
                        row_cells = [""] * max_cols
                        table_matrix.append(row_cells)
                        max_rows += 1
                    if current_col_index < len(table_matrix[current_row_index + i]):
                        table_matrix[current_row_index + i][
                            current_col_index
                        ] = cell_content
                    else:
                        logger.warning(
                            "Failed to apply rowspan cell at row "
                            f"{current_row_index + i}, col {current_col_index}; "
                            "table row is shorter than expected."
                        )
                max_rows = max(current_row_index + row_span, max_rows)
                current_col_index += col_span
            if row_header_flag:
                row_headers_index.append(current_row_index)
            current_row_index += 1

        for i in range(col_header_index_max + 1):
            col_headers_index.append(i)

        if not table_matrix:
            table_matrix = [[]]

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

    def _replace_image_paths(self, content: str, save_name_template: str, tenant_id: str):
        saved_images = []
        
        # Process URL-based images
        url_matches = list(MARKDOWN_IMAGE_PATTERN.finditer(content))
        for match in url_matches:
            full_match = match.group(0)
            image_url = match.group(1)
            should_remove_image = True
            
            if self.image_caption_tool:
                image_file, image_name = get_image_from_url(image_url)
                if image_name:
                    save_image_name = save_name_template.format(image_name)

                    try:
                        self.file_store.write(file=image_file, file_name=image_name, file_path=save_image_name, tenant_id=tenant_id)
                        image_file.seek(0)
                        image_data = image_file.read()
                        image_alt_text = self.image_caption_tool.extract_image(image_data)
                        if image_alt_text:
                            cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
                            
                            image_text = to_markdown_image_text(save_image_name, cleaned_alt)
                            content = content.replace(
                                full_match,
                                image_text,
                            )
                            saved_images.append(save_image_name)
                            logger.info(
                                f"Successfully saved image {save_image_name} from URL: {image_url}"
                            )
                            should_remove_image = False
                    except Exception as ex:
                        logger.warning(f"Failed to save image from URL: {image_url}. {ex}.")

            if should_remove_image:
                content = content.replace(full_match, "")
                logger.warning(f"Failed to save image from URL: {image_url}. Remove image from contents.")
        
        # Process base64 images
        base64_matches = list(BASE64_IMAGE_PATTERN.finditer(content))
        processed_count = 0
        for idx, match in enumerate(base64_matches):
            # Check image count limit
            if processed_count >= MAX_BASE64_IMAGE_COUNT:
                logger.warning(
                    f"Reached max base64 image count ({MAX_BASE64_IMAGE_COUNT}), "
                    f"skipping {len(base64_matches) - processed_count} remaining images"
                )
                break
                
            full_match = match.group(0)
            base64_data = match.group(1)  # e.g., data:image/png;base64,iVBORw0KG...
            
            try:
                # Extract image type and base64 data
                header, base64_string = base64_data.split(',', 1)
                image_type = header.split('/')[1].split(';')[0]  # e.g., 'png', 'jpeg'
                
                # Check size limit before decoding
                estimated_size = len(base64_string) * 3 // 4  # Approximate decoded size
                if estimated_size > MAX_BASE64_IMAGE_SIZE:
                    logger.warning(
                        f"Base64 image size ({estimated_size} bytes) exceeds limit "
                        f"({MAX_BASE64_IMAGE_SIZE} bytes), skipping image {idx}"
                    )
                    content = content.replace(full_match, "")
                    continue
                
                # Decode base64 to bytes
                import base64 as b64
                image_bytes = b64.b64decode(base64_string)
                
                # Generate filename
                image_name = f"embedded_image_{idx}.{image_type}"
                save_image_name = save_name_template.format(image_name)
                
                # Save to file store
                from io import BytesIO
                image_file = BytesIO(image_bytes)
                self.file_store.write(file=image_file, file_name=image_name, file_path=save_image_name, tenant_id=tenant_id)
                
                # Extract caption if tool available
                if self.image_caption_tool:
                    image_alt_text = self.image_caption_tool.extract_image(image_bytes)
                    if image_alt_text:
                        cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
                        image_text = to_markdown_image_text(save_image_name, cleaned_alt)
                        content = content.replace(full_match, image_text)
                        saved_images.append(save_image_name)
                        logger.info(f"Successfully saved base64 image {save_image_name}")
                    else:
                        # No caption, just replace with saved image reference
                        image_text = to_markdown_image_text(save_image_name, "")
                        content = content.replace(full_match, image_text)
                        saved_images.append(save_image_name)
                else:
                    # No caption tool, delete image from content
                    content = content.replace(full_match, "")
                    saved_images.append(save_image_name)
                
                processed_count += 1
                    
            except Exception as ex:
                logger.warning(f"Failed to process base64 image: {ex}")
                # Remove the base64 image from content to avoid bloating
                content = content.replace(full_match, "")

        return content, saved_images

    def read(self, file_item: FileItem) -> List[Document]:
        """Read an HTML file and return a single Markdown document."""
        file_item.file.seek(0)
        html_content = file_item.file.read().decode("utf-8")
        html_content = replace_consecutive_spaces(html_content)

        modified_html, tables = self._extract_tables(html_content)

        markdown_content = markdownify(modified_html)
        for table in tables:
            placeholder = table.attrs.get('data-placeholder')
            if not placeholder:
                logger.warning("Table missing data-placeholder attribute")
                continue

            try:
                table_markdown = self._convert_table_to_markdown(table) + "\n\n"
            except Exception as e:
                logger.warning(f"Failed to convert table to markdown: {e}")
                table_markdown = markdownify(str(table)) + "\n\n"

            if placeholder in markdown_content:
                markdown_content = markdown_content.replace(placeholder, table_markdown)
            else:
                logger.warning(f"Placeholder '{placeholder}' not found in markdown")
                logger.debug(f"Content preview: {markdown_content[:300]}")
        images = []
        markdown_content, images = self._replace_image_paths(
            markdown_content, file_item.kb_id + "/images/{}", tenant_id=file_item.tenant_id,
        )
        logger.info(
            f"Successfully read {file_item.file_name} with images {images}."
        )

        metadata = file_item.metadata()

        docs = [Document(id_=file_item.id, text=markdown_content, metadata=metadata)]
        logger.info(f"Successfully read {file_item.file_name}.")

        return docs
