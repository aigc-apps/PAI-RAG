from io import BytesIO
import os
from typing import List
from pairag.file.models.file_item import FileItem
from pairag.file.readers.base import BaseReader
from pairag.file.store.base import BaseFileStore
from pairag.file.store.oss_store import OssFileStore
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.image_utils import get_image_from_url, markdown_image_text_to_chunk
from pairag.file.utils.text_utils import replace_consecutive_spaces
from llama_index.core.schema import Document
from fastpdf4llm import to_content_list, ProgressInfo
from loguru import logger
import re


def extract_image_path(markdown_text: str) -> str:
    """
    从Markdown图片语法 ![](image_path) 中提取图片路径
    
    Args:
        markdown_text: 包含Markdown图片语法的文本
        content: 包含Markdown图片语法的文本
    Returns:
        str: 提取到的图片路径，如果未找到则返回None
    """
    # 匹配Markdown图片语法的正则表达式
    # ![](path) 或 ![alt](path) 或 ![alt text](path "title")
    pattern = r'!\[.*?\]\((.*?)\)'
    
    match = re.search(pattern, markdown_text)
    if match:
        return match.group(1).strip()  # 返回第一个捕获组（图片路径）
        
    return None


def progress_callback(progress: ProgressInfo):
    logger.info(f"{progress.phase.value}: {progress.current_page}/{progress.total_pages} "
          f"({progress.percentage:.1f}%) - {progress.message}")


class SimplePdfReader(BaseReader):
    def __init__(
        self,
        extract_images: bool = False,
        file_store: BaseFileStore = None,
        image_caption_tool: ImageCaptionTool = None
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        self.extract_images = extract_images
        logger.info("OnlinePdfReader inited.")


    def read(self, file_item: FileItem) -> List[Document]:
        content_list = to_content_list(file_item.file, progress_callback=progress_callback, extract_images=self.extract_images)
        for content in content_list:
            content.text = replace_consecutive_spaces(content.text)

        md_content = "\n\n".join([content.text.rstrip("\n") for content in content_list])

        if self.extract_images and self.image_caption_tool:
            # convert image content to text field
            logger.info(f"[MinerU] replacing image by pattern...")
            save_name_template = file_item.kb_id + "/images/{}"
            for content in content_list:
                content.text = replace_consecutive_spaces(content.text)
                if content.type == "image" and content.text:
                    origin_image_text = content.text
                    image_path = extract_image_path(origin_image_text)
                    image_file, image_name = get_image_from_url(image_path)
                    if image_name:
                        save_image_name = save_name_template.format(image_name)

                        try:
                            upload_result = self.file_store.write(file=image_file, file_name=image_name, file_path=save_image_name, tenant_id=file_item.tenant_id)
                            image_file.seek(0)
                            image_data = image_file.read()
                            image_alt_text = self.image_caption_tool.extract_image(image_data)
                            if image_alt_text:
                                cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
                                content.text = markdown_image_text_to_chunk(upload_result.file_path, cleaned_alt)
                                md_content = md_content.replace(origin_image_text, content.text)
                                logger.info(
                                    f"Successfully saved image {upload_result.file_path} from URL: {image_path}"
                                )
                            else:
                                content.text = content.text.replace(origin_image_text, "")
                                md_content = md_content.replace(origin_image_text, "")
                        except Exception as ex:
                            content.text = content.text.replace(origin_image_text, "")
                            md_content = md_content.replace(origin_image_text, "")
                            logger.exception(
                                f"Failed to save image from URL: {image_path}. Error: {ex}"
                            )
        else:
            content_list = [content for content in content_list if content.type != "image"]

        metadata = file_item.metadata()
        metadata["content_list"] = content_list
        return [
            Document(
                id_=file_item.id,
                text=md_content,
                metadata=metadata,
            )
        ]


if __name__ == "__main__":
    pdf_file = "tests/testdata/pdf_data/pai_document.pdf"
    pdf_file_item = FileItem.from_path(pdf_file, kb_id="test")
    oss_store = OssFileStore(bucket="pai-rag", endpoint="oss-cn-hangzhou.aliyuncs.com")
    pdf_reader = SimplePdfReader(
        extract_images=False,
        file_store=oss_store
    )
    docs = pdf_reader.read(pdf_file_item)
    print(f"Finished with {len(docs[0].metadata['content_list'])} contents.")
