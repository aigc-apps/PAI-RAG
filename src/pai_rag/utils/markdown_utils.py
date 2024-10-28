from PIL import Image
from io import BytesIO
from PIL.PngImagePlugin import PngImageFile
from typing import Any, List, Optional
from llama_index.core.bridge.pydantic import Field, BaseModel
import math

IMAGE_MAX_PIXELS = 512 * 512


class PaiTable(BaseModel):
    data: List[List[str]] = (Field(description="The table data.", default=[]),)
    row_headers_index: Optional[List[int]] = (
        Field(description="The table row headers index.", default=None),
    )
    column_headers_index: Optional[List[int]] = (
        Field(description="The table column headers index.", default=None),
    )

    def get_row_numbers(self):
        return len(self.data)

    def get_col_numbers(self):
        return len(self.data[0])

    def get_row_headers(self):
        if len(self.row_headers_index) == 0:
            return []
        return [self.data[row] for row in self.row_headers_index]

    def get_rows(self):
        if self.row_headers_index:
            data_row_start_index = max(self.row_headers_index) + 1
        else:
            data_row_start_index = 0
        return self.data[data_row_start_index:]

    def get_column_headers(self):
        if len(self.column_headers_index) == 0:
            return []
        return [[row[i] for i in self.column_headers_index] for row in self.data]

    def get_columns(self):
        if self.column_headers_index:
            data_col_start_index = max(self.col_headers_index) + 1
        else:
            data_col_start_index = 0
        return [
            [row[i] for i in range(data_col_start_index, self.get_col_numbers())]
            for row in self.data
        ]


def transform_local_to_oss(oss_cache: Any, image: PngImageFile, doc_name: str) -> str:
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.width <= 50 or image.height <= 50:
            return None

        current_pixels = image.width * image.height

        # 检查像素总数是否超过限制
        if current_pixels > IMAGE_MAX_PIXELS:
            # 计算缩放比例以适应最大像素数
            scale = math.sqrt(IMAGE_MAX_PIXELS / current_pixels)
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)

            # 调整图片大小
            image = image.resize((new_width, new_height), Image.LANCZOS)

        image_stream = BytesIO()
        image.save(image_stream, format="jpeg")

        image_stream.seek(0)
        data = image_stream.getvalue()

        image_url = oss_cache.put_object_if_not_exists(
            data=data,
            file_ext=".jpeg",
            headers={
                "x-oss-object-acl": "public-read"
            },  # set public read to make image accessible
            path_prefix=f"pairag/doc_images/{doc_name.strip()}/",
        )
        print(
            f"Cropped image {image_url} with width={image.width}, height={image.height}."
        )
        return image_url
    except Exception as e:
        print(f"无法打开图片 '{image}': {e}")


def _table_to_markdown(self, table, doc_name):
    markdown = []
    total_cols = max(len(row.cells) for row in table.rows)

    header_row = table.rows[0]
    headers = self._parse_row(header_row, doc_name, total_cols)
    markdown.append("| " + " | ".join(headers) + " |")
    markdown.append("| " + " | ".join(["---"] * total_cols) + " |")

    for row in table.rows[1:]:
        row_cells = self._parse_row(row, doc_name, total_cols)
        markdown.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(markdown)


def convert_table_to_markdown(table: PaiTable, total_cols: int) -> str:
    markdown = []
    if len(table.get_column_headers()) > 0:
        headers = table.get_column_headers()
        rows = table.get_columns()
    else:
        headers = table.get_row_headers()
        rows = table.get_rows()
    if headers:
        for header in headers:
            markdown.append("| " + " | ".join(header) + " |")
        markdown.append("| " + " | ".join(["---"] * total_cols) + " |")
    if rows:
        for row in rows:
            markdown.append("| " + " | ".join(row) + " |")
    return "\n".join(markdown)
