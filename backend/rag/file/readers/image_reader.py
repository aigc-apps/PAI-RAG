from rag.file.readers.base import BaseReader, FileItem, Document, List
from rag.file.store.base import BaseFileStore
from rag.file.store.oss_store import OssFileStore
from rag.file.utils.image_utils import compress_image_if_needed
from rag.file.image_caption_tool import ImageCaptionTool
from loguru import logger


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
        if not isinstance(self.file_store, OssFileStore) or not self.image_caption_tool:
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

            self.file_store.save(image_file, save_image_name)
            image_alt_text = self.image_caption_tool.extract_url(
                self.file_store.get_url(save_image_name)
            )
            image_text = f'<img src="{save_image_name}" alt="{image_alt_text}">'

            metadata = file_item.metadata()
            metadata["images"] = [save_image_name]

            docs = [Document(id_=file_item.id, text=image_text, metadata=metadata)]
            logger.info(f"Successfully read {file_item.file_name}.")

            return docs
        except Exception as e:
            logger.exception(e)
            return []
