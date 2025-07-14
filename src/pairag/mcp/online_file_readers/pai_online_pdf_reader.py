import pymupdf4llm
from loguru import logger
from pairag.file.store.pai_image_store import PaiImageStore
from pairag.mcp.online_file_readers.utils.image_utils import (
    replace_markdown_images_with_oss_urls,
)
from typing import Dict, List, Optional, Union
from pathlib import Path
from llama_index.core.schema import Document
import traceback
import tempfile
import fitz
import os
from llama_index.core.readers.base import BaseReader


class PaiOnlinePDFReader(BaseReader):
    def __init__(
        self,
        image_store: PaiImageStore = None,
    ) -> None:
        self.image_store = image_store

    def parse_pdf(
        self,
        pdf_path: str,
    ):
        """
        执行从 pdf 转换到 md 的过程，输出 md 文件到 pdf 文件所在的目录

        :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
        """
        try:
            pdf_name = os.path.basename(pdf_path).split(".")[0]
            pdf_name = pdf_name.replace(" ", "_")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, pdf_name)
                if self.image_store:
                    md_content = pymupdf4llm.to_markdown(
                        pdf_path, write_images=True, image_path=temp_file_path
                    )
                    md_content = replace_markdown_images_with_oss_urls(
                        pdf_path, self.image_store, pdf_name
                    )
                else:
                    md_content = pymupdf4llm.to_markdown(pdf_path, write_images=False)

            return md_content

        except Exception:
            logger.error(traceback.format_exc())
            raise

    def load_data(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format."""
        return self.load(file_path, metadata=metadata, extra_info=extra_info)

    def load(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): file path of PDF file (accepts string or Path).
            metadata (bool, optional): if metadata to be included or not. Defaults to True.
            extra_info (Optional[Dict], optional): extra information related to each document in dict format. Defaults to None.

        Raises:
            TypeError: if extra_info is not a dictionary.
            TypeError: if file_path is not a string or Path.

        Returns:
            List[Document]: list of documents.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        pdf_name = os.path.basename(file_path).split(".")[0]
        md_content = self.parse_pdf(file_path)
        pdf_page_num = len(fitz.open(file_path))
        logger.info(
            f"[PaiOnlinePDFReader] successfully processed pdf file {file_path}."
        )
        metadata = {"page_num": pdf_page_num, "file_name": pdf_name}
        if extra_info is not None:
            metadata.update(extra_info)
        docs = []
        doc = Document(text=md_content, extra_info=extra_info)
        docs.append(doc)
        logger.info(f"[PaiOnlinePDFReader] successfully loaded {len(docs)} nodes.")
        return docs
