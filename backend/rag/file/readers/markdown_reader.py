import re
from typing import List
from rag.file.models.file_item import FileItem
from rag.file.readers.base import BaseReader
from rag.file.store.base import BaseFileStore
from rag.file.store.oss_store import OssFileStore

from rag.file.utils.image_utils import get_image_from_url
from llama_index.core.schema import Document
from loguru import logger

from rag.file.image_caption_tool import ImageCaptionTool


REGEX_H1 = "===+"
REGEX_H2 = "---+"
REGEX_USELESS_PHRASE = "\{#[0-9a-z]+\}"  # Only for aliyun docs
MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\(((?!https?://|www\.)[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))\)",
    re.IGNORECASE,
)
HTML_IMAGE_PATTERN = re.compile(
    r'<img[^>]*src=["\']((?!https?://|www\.)[^"\']+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff))["\'][^>]*>',
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

            image_file, image_name = get_image_from_url(local_url)
            if image_name:
                save_image_name = save_name_template.format(image_name)

                try:
                    self.file_store.save(image_file, save_image_name)
                    image_alt_text = self.image_caption_tool.extract_url(
                        self.file_store.get_url(save_image_name)
                    )
                    image_text = f'<img src="{save_image_name}" alt="{image_alt_text}">'
                    content = content.replace(full_match, image_text)
                    saved_images.append(save_image_name)

                    logger.info(
                        f"Successfully saved image {save_image_name} from URL: {local_url}"
                    )
                except Exception as ex:
                    logger.exception(
                        f"Failed to save image from URL: {local_url}. Error: {ex}"
                    )
        return content, saved_images

    def read(self, file_item: FileItem) -> List[Document]:
        md_content = ""
        pre_line = ""
        file_item.file.seek(0)
        while True:
            line = file_item.file.readline().decode("utf-8")
            if not line:
                break
            is_code = False
            striped_line = re.sub(REGEX_USELESS_PHRASE, "", line)
            if striped_line.startswith("```"):
                is_code = not is_code

            if not striped_line:
                md_content += pre_line
                pre_line = "\n"
            elif re.match(REGEX_H1, striped_line):
                md_content += f"# {pre_line}"
                pre_line = ""
            elif re.match(REGEX_H2, striped_line):
                md_content += f"## {pre_line}"
                pre_line = ""
            else:
                md_content += pre_line
                pre_line = striped_line
                if is_code or line.startswith("#") or line.endswith("  \n"):
                    pre_line = f"{striped_line}\n"

        md_content += pre_line

        if isinstance(self.file_store, OssFileStore) and self.image_caption_tool:
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
