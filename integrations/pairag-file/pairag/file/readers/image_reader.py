from io import BytesIO
from typing import BinaryIO
from pairag.file.readers.base import BaseReader, FileItem, Document, List
from pairag.file.store.base import BaseFileStore
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.image_utils import to_markdown_image_text
from pairag.file.utils.text_utils import replace_consecutive_spaces
from PIL import Image
from pillow_heif import register_heif_opener

from loguru import logger
import re

register_heif_opener()

def convert_heif_to_jpeg(image_file: BinaryIO) -> BinaryIO:
    jpg_image_file = BytesIO()
    image = Image.open(image_file)
    image.convert("RGB").save(jpg_image_file, "JPEG", quality=90)
    jpg_image_file.seek(0)
    return jpg_image_file

class ImageReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, image_caption_tool: ImageCaptionTool = None
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        logger.info("ImageReader inited.")

    def read(self, file_item: FileItem) -> List[Document]:
        """
        Read a CSV file and return a list of Documents.
        """
        docs = []
        if not self.image_caption_tool:
            logger.warning(
                "Will not parse image files when image caption model is not configured."
            )
            return docs
        try:
            file_item.file.seek(0)

            image_file = file_item.file
            if file_item.file_extension.lower() == ".heic":
                image_file = convert_heif_to_jpeg(image_file)
                file_item.file_extension = ".jpg"
            if not image_file:
                return docs

            save_image_name = f"{file_item.kb_id}/images/{file_item.file_md5}" + file_item.file_extension

            upload_result = self.file_store.write(file=image_file, file_name=file_item.file_name, file_path=save_image_name, tenant_id=file_item.tenant_id)
            image_file.seek(0)
            image_data = image_file.read()
            image_alt_text = self.image_caption_tool.extract_image(image_data, file_extension=file_item.file_extension)
            image_alt_text = replace_consecutive_spaces(image_alt_text)
            if image_alt_text:
                cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
                image_text = to_markdown_image_text(upload_result.file_path, cleaned_alt)

                metadata = file_item.metadata()
                docs.append(Document(id_=file_item.id, text=image_text, metadata=metadata))
            logger.info(f"Successfully read {file_item.file_name}.")

            return docs
        except Exception as e:
            logger.exception(e)
            return docs
