from pairag.file.readers.base import BaseReader, FileItem, Document, List
from pairag.file.store.base import BaseFileStore
from pairag.file.utils.video_caption_tool import VideoCaptionTool
from pairag.file.utils.text_utils import replace_consecutive_spaces
from loguru import logger


class VideoReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, video_caption_tool: VideoCaptionTool = None
    ):
        self.file_store = file_store
        self.video_caption_tool = video_caption_tool
        logger.info("VideoReader inited.")

    def read(self, file_item: FileItem) -> List[Document]:
        """
        Read a CSV file and return a list of Documents.
        """
        docs = []
        if not self.video_caption_tool:
            logger.warning(
                "Will not parse video files when video store is not configured."
            )
            return docs
        try:
            file_item.file.seek(0)

            if not file_item.file:
                return docs

            video_data = file_item.file.read()
            video_alt_text = self.video_caption_tool.extract_video(video_data, file_item.file_extension)
            video_alt_text = replace_consecutive_spaces(video_alt_text)
            if video_alt_text:
                metadata = file_item.metadata()
                docs.append(Document(id_=file_item.id, text=video_alt_text, metadata=metadata))
            logger.info(f"Successfully read {file_item.file_name}.")

            return docs
        except Exception as e:
            logger.exception(e)
            return docs
