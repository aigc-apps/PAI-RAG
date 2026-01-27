import aiohttp
from loguru import logger
import os
import asyncio
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import tempfile
import json
import re
import io
import hashlib
from markdownify import markdownify
from fastpdf4llm import ContentBlock
from pairag.file.readers.base import BaseReader
from pairag.file.utils.font_utils import infer_mineru_api_title_level
from pairag.file.utils.image_utils import to_markdown_image_text
from pairag.file.utils.mineru_utils import make_content_block
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.store.base import BaseFileStore
from pairag.file.models.file_item import FileItem
from llama_index.core.schema import Document
from pairag.file.utils.async_helper import run_sync
from pairag.file.utils.text_utils import replace_consecutive_spaces


def sanitize_filename(filename: str) -> str:
    """
    使用文件名计算hash值，避免所有特殊字符问题。
    保留原始文件的扩展名。
    """
    # 分离文件名和扩展名
    name, ext = os.path.splitext(filename)
    
    # 计算文件名的hash值（使用SHA256，取前16位）
    hash_obj = hashlib.sha256(filename.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:16]
    
    # 如果有扩展名，保留它；否则返回hash值
    if ext:
        return f"{hash_hex}{ext}"
    else:
        return hash_hex


def extract_api_result(
    zip_path: str,
    extract_to: str,
    root_dir: str,
    image_store: BaseFileStore = None,
    save_image_template: str = None,
    image_caption_tool: ImageCaptionTool = None,
    tenant_id: str = None,
    ) -> Tuple[str, List[ContentBlock]]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    for file in os.listdir(extract_to):
        logger.info(f"Extracted file: {file}, path: {os.path.join(extract_to, file)}")
    md_file_path = os.path.join(extract_to, root_dir, f"{root_dir}.md")
    md_content = ""
    try:
        with open(md_file_path, 'r') as f:
            md_content = f.read()
    except Exception as e:
        logger.error(f"Error extracting md content from {md_file_path}: {e}")
        raise

    content_list = []
    image_local_dir = os.path.join(extract_to, root_dir, "images")
    middle_json_path = os.path.join(extract_to, root_dir, f"{root_dir}_middle.json")
    try:
        with open(middle_json_path, 'r') as f:
            middle_json = json.load(f)
            pages_objects = middle_json.get("pdf_info", [])
            for page in pages_objects:
                page_idx = page.get("page_idx", 0) + 1
                page_blocks = page.get("preproc_blocks", [])
                for block in page_blocks:
                    content_block = make_content_block(block, page_idx, image_store, image_local_dir, save_image_template, image_caption_tool, tenant_id)
                    if content_block:
                        content_list.append(content_block)
            logger.info(f"Extracted content list from {middle_json_path}, length: {len(content_list)}")
            return md_content, content_list
    except Exception as e:
        logger.error(f"Error extracting content list from {middle_json_path}: {e}")
        raise


class MineruPdfReader(BaseReader):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        model_version: str = "pipeline",
        file_store: BaseFileStore = None,
        image_caption_tool: ImageCaptionTool = None
    ):
        self.endpoint = endpoint or os.environ["MINERU_ENDPOINT"]
        self.endpoint = self.endpoint.rstrip("/")
        self.token = token or os.environ["MINERU_TOKEN"]

        assert self.endpoint and self.token, "MineruPdfReader endpoint and token are required"

        self.model_version = model_version
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool

        self.image_store = None
        if self.image_caption_tool:
            self.image_store = self.file_store
        logger.info(f"MineruPdfReader: init with endpoint {self.endpoint}, token {self.token}, model version {self.model_version}")


    def read(self, file_item: FileItem) -> List[Document]:
        save_image_template = file_item.kb_id + "/images/{}"
        md_content, content_list = run_sync(
            self.parse_file_async(
                file=file_item.file,
                tenant_id=file_item.tenant_id,
                pdf_file_name=sanitize_filename(file_item.file_name),
                save_image_template=save_image_template,
                method="auto",
                backend="pipeline",
                lang="ch",
            )
        )
        md_content = replace_consecutive_spaces(md_content)
        for content in content_list:
            content.text = replace_consecutive_spaces(content.text)
        metadata = file_item.metadata()
        metadata["content_list"] = content_list

        # save_md_file_name = os.path.join(file_item.kb_id, "markdown", file_item.file_name + ".md")

        return [Document(id_=file_item.id, text=md_content, metadata=metadata)]

    async def parse_file_async(
        self,
        file,
        tenant_id: str,
        pdf_file_name: str,
        save_image_template: str,
        method: str = "auto",
        backend: str = "pipeline",
        lang: Optional[str] = None,
        callback: Optional[Callable] = None,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_zip_path = os.path.join(temp_dir, "output.zip")
            file_base_name = os.path.splitext(pdf_file_name)[0]
            output_path = os.path.join(temp_dir, file_base_name, method)
            os.makedirs(output_path, exist_ok=True)

            # 使用 aiohttp.FormData 来上传文件
            # 支持 file-like object (BinaryIO) 和 bytes
            # 如果是 file-like object，确保文件指针在开始位置
            if hasattr(file, 'read') and hasattr(file, 'seek'):
                # file-like object (BinaryIO)
                file.seek(0)  # 确保文件指针在开始位置
                file_data = file
            elif isinstance(file, bytes):
                # bytes 数据，转换为 BytesIO
                file_data = io.BytesIO(file)
            else:
                # 其他情况，尝试直接使用
                file_data = file
            
            form_data = aiohttp.FormData()
            form_data.add_field('files', file_data, filename=pdf_file_name, content_type='application/pdf')
            
            # 添加其他表单字段
            form_data.add_field('lang_list', lang if lang else 'ch')
            form_data.add_field('backend', backend)
            form_data.add_field('parse_method', method)
            form_data.add_field('formula_enable', 'true')
            form_data.add_field('table_enable', 'true')
            form_data.add_field('return_md', 'true')
            form_data.add_field('return_middle_json', 'true')
            form_data.add_field('return_model_output', 'true')
            form_data.add_field('return_content_list', 'true')
            form_data.add_field('return_images', 'true')
            form_data.add_field('response_format_zip', 'true')
            form_data.add_field('start_page_id', '0')
            form_data.add_field('end_page_id', '99999')

            headers = {"Accept": "application/json", "Authorization": f"Bearer {self.token}"}
            md_content = ""
            content_list = []

            try:
                logger.info(f"[MinerU] invoke api: {self.endpoint}/file_parse")

                timeout = aiohttp.ClientTimeout(total=1800)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url=f"{self.endpoint}/file_parse", data=form_data, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"[MinerU] api failed with status {response.status}, error: {await response.text()}")
                            raise RuntimeError(f"[MinerU] api failed with status {response.status}, error: {await response.text()}")
                        
                        if response.headers.get("Content-Type") == "application/zip":
                            logger.info(f"[MinerU] zip file returned, saving to {output_zip_path}...")
                            with open(output_zip_path, "wb") as f:
                                f.write(await response.read())

                        else:
                            logger.warning(f"[MinerU] not zip returned from api: {response.headers.get('Content-Type')}")
            except Exception as e:
                raise RuntimeError(f"[MinerU] api failed with exception {e}")

            try:
                logger.info(f"[MinerU] Unzip to {output_path}...")
                md_content, content_list = extract_api_result(output_zip_path, output_path, file_base_name, self.image_store, save_image_template, self.image_caption_tool, tenant_id)
            except Exception as e:
                logger.error(f"[MinerU] error extracting content list from {output_zip_path}: {e}")
                raise
            
            try:
                logger.info(f"[MinerU] inferring title level from content list...")
                infer_mineru_api_title_level(content_list)
            except Exception as e:
                logger.error(f"[MinerU] error inferring title level from content list: {e}")
                raise

            return md_content, content_list


async def main():
    parser = MineruPdfReader(endpoint=os.environ["MINERU_ENDPOINT"], token=os.environ["MINERU_TOKEN"])
    input_path = "tests/testdata/pdf_data/iPhone 16.pdf"
    with open(input_path, 'rb') as f:
        file = f.read()
        md_content, content_list = await parser.parse_file_async(file=file, pdf_file_name="test.pdf", save_image_template="test/images/{}", method="auto", backend="pipeline", lang="ch")

        with open("test.json", "w") as f:
            json.dump([content.model_dump(exclude_none=True) for content in content_list], f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())