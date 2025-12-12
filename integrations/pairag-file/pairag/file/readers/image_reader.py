from pairag.file.readers.base import BaseReader, FileItem, Document, List
from pairag.file.store.base import BaseFileStore
from pairag.file.utils.image_utils import compress_image_if_needed
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.image_utils import to_markdown_image_text
from loguru import logger
import re


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
        if not self.image_caption_tool:
            logger.warning(
                "Will not parse image files when image store is not configured."
            )
            return []
        try:
            file_item.file.seek(0)
            save_image_name = f"{file_item.kb_id}/images/{file_item.file_md5}.jpeg"

            image_file = compress_image_if_needed(file_item.file)
            if not image_file:
                return []

            upload_result = self.file_store.write(file=image_file, file_name=file_item.file_name, file_path=save_image_name, tenant_id=file_item.tenant_id)
            image_file.seek(0)
            image_data = image_file.read()
            image_alt_text = self.image_caption_tool.extract_image(image_data)
            cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
            image_text = to_markdown_image_text(upload_result.file_path, cleaned_alt)

            metadata = file_item.metadata()

            docs = [Document(id_=file_item.id, text=image_text, metadata=metadata)]
            logger.info(f"Successfully read {file_item.file_name}.")

            return docs
        except Exception as e:
            logger.exception(e)
            return []
