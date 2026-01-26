import re
from typing import List
from pairag.file.models.file_item import FileItem
from pairag.file.readers.base import BaseReader
from pairag.file.store.base import BaseFileStore

from pairag.file.utils.image_utils import get_image_from_url
from llama_index.core.schema import Document
from pairag.file.utils.image_utils import to_markdown_image_text
from pairag.file.utils.text_utils import replace_consecutive_spaces
from loguru import logger

from pairag.file.utils.image_caption_tool import ImageCaptionTool


REGEX_H1 = "===+"
REGEX_H2 = "---+"
MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\((https?://[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff)(?:\?[^\s)]*)?)\)",
    re.IGNORECASE,
)
HTML_IMAGE_PATTERN = re.compile(
    r'<img[^>]*src=["\'](https?://[^\s)]+\.(?:png|jpe?g|gif|bmp|svg|webp|tiff)(?:\?[^\s"\']*)?)["\'][^>]*>',
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
        self, content: str, pattern: re.Pattern, save_name_template: str, tenant_id: str
    ):
        image_matches = pattern.finditer(content)
        saved_images = []
        for match in image_matches:
            full_match = match.group(0)  # 整个匹配
            local_url = match.group(1)  # 捕获的URL
            should_remove_image = True
            if self.image_caption_tool:
                image_file, image_name = get_image_from_url(local_url)
                if image_name:
                    save_image_name = save_name_template.format(image_name)

                    try:
                        upload_result =self.file_store.write(file=image_file, file_name=image_name, file_path=save_image_name, tenant_id=tenant_id)
                        image_file.seek(0)
                        image_data = image_file.read()
                        image_alt_text = self.image_caption_tool.extract_image(image_data)
                        if image_alt_text:
                            cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
                            image_text = to_markdown_image_text(upload_result.file_path, cleaned_alt)
                            content = content.replace(full_match, image_text)
                            saved_images.append(upload_result.file_path)

                            logger.info(
                                f"Successfully saved image {upload_result.file_path} from URL: {local_url}"
                            )
                            should_remove_image = False
                    except Exception as ex:
                        logger.warning(f"Failed to save image from URL: {local_url}. {ex}.")
            
            if should_remove_image:
                content = content.replace(full_match, "")
                logger.warning(f"Failed to save image from URL: {local_url}. Remove image from contents.")
        return content, saved_images

    def read(self, file_item: FileItem) -> List[Document]:
        file_item.file.seek(0)

        md_content = file_item.file.read().decode("utf-8")
        md_content = replace_consecutive_spaces(md_content)

        md_content, _ = self.replace_image_by_pattern(
            md_content, MARKDOWN_IMAGE_PATTERN, file_item.kb_id + "/images/{}", tenant_id=file_item.tenant_id,
        )
        md_content, _ = self.replace_image_by_pattern(
            md_content, HTML_IMAGE_PATTERN, file_item.kb_id + "/images/{}", tenant_id=file_item.tenant_id,
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
    md_reader = MarkdownReader()
    doc = md_reader.read(md_file_item)
    print(doc[0].text)
    print(doc[0].metadata["images"])
    print("finished.")
