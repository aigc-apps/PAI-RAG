import re
from typing import List
from pairag.file.models.file_item import FileItem
from pairag.file.readers.base import BaseReader
from pairag.file.store.base import BaseFileStore
from pairag.file.store.oss_store import OssFileStore

from pairag.file.utils.image_utils import get_image_from_url
from llama_index.core.schema import Document
from pairag.file.utils.image_utils import to_markdown_image_text
from loguru import logger

from pairag.file.utils.image_caption_tool import ImageCaptionTool


REGEX_H1 = "===+"
REGEX_H2 = "---+"
MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\((https?://[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))\)",
    re.IGNORECASE,
)
HTML_IMAGE_PATTERN = re.compile(
    r'<img[^>]*src=["\'](https?://[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))["\'][^>]*>',
    re.IGNORECASE,
)


class MarkdownReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, image_caption_tool: ImageCaptionTool = None
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        logger.info("MarkdownReader inited.")

    def replace_image_by_pattern(
        self, content: str, pattern: re.Pattern, save_name_template: str
    ):
        image_matches = pattern.finditer(content)
        saved_images = []
        for match in image_matches:
            full_match = match.group(0)  # 整个匹配
            local_url = match.group(1)  # 捕获的URL
            if self.image_caption_tool and isinstance(self.file_store, OssFileStore):
                image_file, image_name = get_image_from_url(local_url)
                if image_name:
                    save_image_name = save_name_template.format(image_name)

                    try:
                        self.file_store.save(image_file, save_image_name)
                        image_alt_text = self.image_caption_tool.extract_url(
                            self.file_store.get_url(save_image_name)
                        )
                        cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
                        image_text = to_markdown_image_text(save_image_name, cleaned_alt)
                        content = content.replace(full_match, image_text)
                        saved_images.append(save_image_name)

                        logger.info(
                            f"Successfully saved image {save_image_name} from URL: {local_url}"
                        )
                    except Exception as ex:
                        logger.exception(
                            f"Failed to save image from URL: {local_url}. Error: {ex}"
                        )
            else:
                content = content.replace(full_match, "") # 移除图片链接
        return content, saved_images

    def read(self, file_item: FileItem) -> List[Document]:
        file_item.file.seek(0)

        md_content = file_item.file.read().decode("utf-8")

        md_content, _ = self.replace_image_by_pattern(
            md_content, MARKDOWN_IMAGE_PATTERN, file_item.kb_id + "/images/{}"
        )
        md_content, _ = self.replace_image_by_pattern(
            md_content, HTML_IMAGE_PATTERN, file_item.kb_id + "/images/{}"
        )

        logger.info(
            f"[MarkdownReader] successfully processed markdown file {file_item.file_name}."
        )
        docs = []
        metadata = file_item.metadata()
        doc = Document(id_=file_item.id, text=md_content, extra_info=metadata)
        docs.append(doc)
        logger.info(
            f"[PaiMarkdownReader] successfully loaded {len(docs)} nodes from {file_item.file_name}."
        )
        return docs


if __name__ == "__main__":
    md_file = "tests/testdata/pai_document.md"
    md_file_item = FileItem.from_path(md_file, knowledgebase_id="test")
    oss_store = OssFileStore(bucket="pai-rag", endpoint="oss-cn-hangzhou.aliyuncs.com")
    md_reader = MarkdownReader(file_store=oss_store)
    doc = md_reader.read(md_file_item)
    print(doc[0].text)
    print(doc[0].metadata["images"])
    print("finished.")
