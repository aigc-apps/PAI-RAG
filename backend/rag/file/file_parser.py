import uuid
from loguru import logger
from typing import List
import copy
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

from db.models.knowledgebase.knowledgebase import ChunkConfig, KbEntity
from rag.file.models.file_item import FileItem
from rag.file.readers.base import BaseReader
from rag.file.readers.csv_reader import CsvReader
from rag.file.readers.doc_reader import DocxReader
from rag.file.readers.excel_reader import ExcelReader
from rag.file.readers.html_reader import HtmlReader
from rag.file.readers.image_reader import ImageReader
from rag.file.readers.jsonl_reader import JsonReader
from rag.file.readers.markdown_reader import MarkdownReader
from rag.file.readers.pdf_reader import MineruPdfReader
from rag.file.readers.pptx_reader import PptxReader
from rag.file.readers.text_reader import TextReader
from rag.file.readers.online_pdf_reader import OnlinePdfReader
from rag.file.splitters.pai_markdown_parser import MarkdownNodeParser
from rag.file.store.base import BaseFileStore
from rag.file.image_caption_tool import ImageCaptionTool


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


def node_id_func(i: int, doc: BaseNode) -> str:
    return uuid.uuid4().hex


class FileParser:
    def __init__(
        self,
        file_store: BaseFileStore,
        knowledgebase: KbEntity,
        image_caption_tool: ImageCaptionTool = None,
    ):
        self.file_store = file_store
        self.knowledgebase = knowledgebase
        self.image_caption_tool = image_caption_tool

    def _get_reader(self, file_extension: str, is_attachment: bool=False) -> BaseReader:
        if is_attachment:
            match file_extension:
                case ".docx":
                    return DocxReader(file_store=self.file_store)
                case ".pdf":
                    return OnlinePdfReader(file_store=self.file_store)
                case ".md":
                    return MarkdownNodeParser(file_store=self.file_store)
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
                    return ExcelReader()
                case ".xls":
                    return ExcelReader()
                case ".csv":
                    return CsvReader()
                case ".jsonl":
                    return JsonReader()
                case _:
                    raise ValueError(f"不支持的文件类型: {file_extension}")

    # 读取文件解析为Document列表
    def read_file(self, file_item: FileItem, is_attachment: bool) -> List[Document]:
        reader = self._get_reader(file_item.file_extension, is_attachment=is_attachment)
        return reader.read(file_item)

    def split_docs(
        self, docs: List[Document], chunk_config: ChunkConfig
    ) -> List[BaseNode]:
        splitted_nodes: List[BaseNode] = []

        for doc_node in docs:
            original_metadata = copy.deepcopy(doc_node.metadata)
            logger.info(f"Start splitting document: {doc_node.metadata['file_name']} with id {doc_node.id_}")

            chunks = []
            doc_type = doc_node.metadata["file_extension"]
            if doc_type in IMAGE_DOC_TYPES:
                node_id = uuid.uuid4().hex
                chunks.append(
                    TextNode(
                        id_=node_id,
                        text=doc_node.text,
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
                        chunk_overlap_size=chunk_config.chunk_overlap,
                        base_parser=parser,
                    )
                    chunks = md_node_parser.get_nodes_from_documents([doc_node])
                else:
                    # txt格式等纯文本
                    chunks = parser.get_nodes_from_documents([doc_node])

            for chunk in chunks:
                chunk.metadata = copy.deepcopy(original_metadata)
                logger.info(f"chunk.metadata {chunk.metadata}")
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
        docs = self.read_file(file_item, is_attachment)
        chunk_config = ChunkConfig.model_validate(self.knowledgebase.chunk_config)
        nodes = self.split_docs(docs, chunk_config=chunk_config)
        return docs, nodes
