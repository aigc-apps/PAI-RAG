import uuid
from loguru import logger
from typing import List
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter
from pairag.file.models.file_item import FileItem
from pairag.file.readers.base import BaseReader
from pairag.file.readers.csv2md_reader import Csv2MdReader
from pairag.file.readers.doc_reader import DocxReader
from pairag.file.readers.excel2md_reader import Excel2MdReader
from pairag.file.readers.html_reader import HtmlReader
from pairag.file.readers.image_reader import ImageReader
from pairag.file.readers.jsonl2md_reader import Json2MdReader
from pairag.file.readers.markdown_reader import MarkdownReader
from pairag.file.readers.pdf_reader import MineruPdfReader
from pairag.file.readers.pptx_reader import PptxReader
from pairag.file.readers.text_reader import TextReader
from pairag.file.readers.online_pdf_reader import OnlinePdfReader
from pairag.file.nodeparsers.pai_markdown_parser import MarkdownNodeParser
from pairag.file.store.base import BaseFileStore
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SENTENCE_SEPARATOR,
    DEFAULT_PARSER_TYPE,
)
from pairag.file.utils.image_utils import MARKDOWN_IMAGE_PATTERN, markdown_image_text_to_chunk
from llama_index.core.bridge.pydantic import Field, BaseModel
from typing import Optional
import re


IMAGE_DOC_TYPES = set([".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"])
DOC_TYPES_DO_NOT_NEED_CHUNKING = set([".csv", ".xlsx", ".xls", ".jsonl"])
DOC_TYPES_CONVERT_TO_MD = set([".md", ".pdf", ".docx", ".htm", ".html", ".pptx"])
DEFAULT_EXCLUDED_METADATA_KEYS = [
    "file_type",
    "file_size",
    "creation_date",
    "last_modified_date",
    "last_accessed_date",
    "file_path",
    "image_url",
    "total_pages",
    "source",
    "row_number",
    "image_info_list",
    "file_url",
    "ref_doc_id",
]



class ChunkConfig(BaseModel):
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP)
    parser_type: str = Field(default=DEFAULT_PARSER_TYPE)
    separator: str = Field(default=DEFAULT_SENTENCE_SEPARATOR)


def node_id_func(i: int, doc: BaseNode) -> str:
    return uuid.uuid4().hex


class FileParser:
    def __init__(
        self,
        file_store: BaseFileStore,
        image_caption_tool: ImageCaptionTool = None,
        chunk_config: Optional[ChunkConfig] = ChunkConfig(),
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        self.chunk_config = chunk_config

    def _get_reader(
            self,
            file_extension: str,
            is_attachment: bool=False,
            chunk_size: int=DEFAULT_CHUNK_SIZE) -> BaseReader:
        if is_attachment:
            match file_extension:
                case ".docx":
                    return DocxReader(file_store=self.file_store)
                case ".pdf":
                    return OnlinePdfReader(file_store=self.file_store)
                case ".md":
                    return MarkdownReader(file_store=self.file_store)
                case ".txt":
                    return TextReader()
                case ".jpg":
                    return ImageReader(
                        file_store=self.file_store,
                    )
                case ".png":
                    return ImageReader(
                        file_store=self.file_store,
                    )
                case ".jpeg":
                    return ImageReader(
                        file_store=self.file_store,
                    )
                case ".xlsx":
                    return Excel2MdReader(chunk_size=chunk_size)
                case ".xls":
                    return Excel2MdReader(chunk_size=chunk_size)
                case ".pptx":
                    return PptxReader(
                        file_store=self.file_store,
                    )
                case ".csv":
                    return Csv2MdReader(chunk_size=chunk_size)
                case ".jsonl":
                    return Json2MdReader(chunk_size=chunk_size)
                case _:
                    raise ValueError(f"不支持的附件文件类型: {file_extension}")
        else:
            match file_extension:
                case ".md":
                    return MarkdownReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".docx":
                    return DocxReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".pptx":
                    return PptxReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".pdf":
                    return MineruPdfReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".htm":
                    return HtmlReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".html":
                    return HtmlReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".jpg":
                    return ImageReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".png":
                    return ImageReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".jpeg":
                    return ImageReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".xlsx":
                    return Excel2MdReader(chunk_size=chunk_size)
                case ".xls":
                    return Excel2MdReader(chunk_size=chunk_size)
                case ".csv":
                    return Csv2MdReader(chunk_size=chunk_size)
                case ".jsonl":
                    return Json2MdReader(chunk_size=chunk_size)
                case ".txt":
                    return TextReader()
                case _:
                    raise ValueError(f"不支持的文件类型: {file_extension}")

    # 读取文件解析为Document列表
    def read_file(
            self,
            file_item: FileItem,
            is_attachment: bool,
            chunk_config: ChunkConfig = None,
        ) -> List[Document]:
        reader = self._get_reader(
            file_item.file_extension,
            is_attachment=is_attachment,
            chunk_size=chunk_config.chunk_size if chunk_config else DEFAULT_CHUNK_SIZE)
        return reader.read(file_item)

    def split_docs(
        self, docs: List[Document], chunk_config: ChunkConfig
    ) -> List[BaseNode]:
        splitted_nodes: List[BaseNode] = []

        for doc_node in docs:
            logger.info(f"Start splitting document: {doc_node.metadata['file_name']} with id {doc_node.id_}")

            chunks = []
            doc_type = doc_node.metadata["file_extension"]
            if doc_type in IMAGE_DOC_TYPES:
                node_id = uuid.uuid4().hex
                match = re.fullmatch(MARKDOWN_IMAGE_PATTERN, doc_node.text.strip())
                alt, image_url = match.group(1), match.group(2)
                chunk_text = markdown_image_text_to_chunk(image_url, alt)
                chunks.append(
                    TextNode(
                        id_=node_id,
                        text=chunk_text,
                    )
                )

            elif doc_type in DOC_TYPES_DO_NOT_NEED_CHUNKING:
                # 表格格式文档
                node_id = uuid.uuid4().hex
                chunks.append(
                    TextNode(
                        id_=node_id,
                        text=doc_node.text,
                    )
                )
            else:
                parser = SentenceSplitter(
                    id_func=node_id_func,
                    chunk_size=chunk_config.chunk_size,
                    chunk_overlap=chunk_config.chunk_overlap,
                    paragraph_separator=chunk_config.separator,
                    include_metadata=False,
                )
                if doc_type in DOC_TYPES_CONVERT_TO_MD:
                    # markdown格式(pdf, md, html, doc 等)
                    md_node_parser = MarkdownNodeParser(
                        chunk_size=chunk_config.chunk_size,
                        chunk_overlap=chunk_config.chunk_overlap,
                        id_func=node_id_func,
                    )
                    chunks = md_node_parser.get_nodes_from_documents([doc_node])
                else:
                    # txt格式等纯文本
                    chunks = parser.get_nodes_from_documents([doc_node])

            for chunk in chunks:
                chunk.metadata = doc_node.metadata
                chunk.metadata["doc_id"] = doc_node.id_
                chunk.relationships = {
                            NodeRelationship.SOURCE: RelatedNodeInfo(
                                node_id=doc_node.id_, metadata={}
                            ),
                        }
            splitted_nodes.extend(chunks)
            logger.info(
                f"Finished split document into {len(chunks)} chunks: {doc_node.metadata['file_name']}"
            )

        for node in splitted_nodes:
            node.excluded_embed_metadata_keys = list(
                set(node.excluded_embed_metadata_keys + DEFAULT_EXCLUDED_METADATA_KEYS)
            )
            node.excluded_llm_metadata_keys = list(
                set(node.excluded_llm_metadata_keys + DEFAULT_EXCLUDED_METADATA_KEYS)
            )

        logger.info(
            f"[DataReader] Split {len(docs)} documents into {len(splitted_nodes)} nodes."
        )

        return splitted_nodes

    def parse(self, file_item: FileItem, is_attachment: bool):
        docs = self.read_file(file_item, is_attachment, chunk_config=self.chunk_config)
        nodes = self.split_docs(docs, chunk_config=self.chunk_config)
        return docs, nodes
