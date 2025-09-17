from io import BytesIO
import os
from typing import List
from pairag.file.models.file_item import FileItem
from pairag.file.readers.base import BaseReader
from pairag.file.store.base import BaseFileStore
from pairag.file.store.oss_store import OssFileStore
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from llama_index.core.schema import Document
import pymupdf4llm
import pymupdf

from loguru import logger

class OnlinePdfReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, image_caption_tool: ImageCaptionTool = None
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        logger.info("OnlinePdfReader inited.")

    def read(self, file_item: FileItem) -> List[Document]:
        md_content = pymupdf4llm.to_markdown(pymupdf.Document(stream=file_item.file), write_images=False)
        save_md_file_name = os.path.join(
            file_item.kb_id, "markdown", file_item.file_name + ".md"
        )
        self.file_store.save(BytesIO(md_content.encode("utf-8")), save_md_file_name)

        metadata = file_item.metadata()
        return [
            Document(
                id_=file_item.id,
                text=md_content,
                metadata=metadata,
            )
        ]


if __name__ == "__main__":
    pdf_file = "tmp/test.pdf"
    pdf_file_item = FileItem.from_path(pdf_file, kb_id="test")
    oss_store = OssFileStore(bucket="pai-rag", endpoint="oss-cn-hangzhou.aliyuncs.com")
    pdf_reader = OnlinePdfReader(
        file_store=oss_store
    )
    doc = pdf_reader.read(pdf_file_item)
    print("finished.")
