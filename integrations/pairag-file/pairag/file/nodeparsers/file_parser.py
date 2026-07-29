import uuid
from loguru import logger
from typing import List
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from pairag.file.nodeparsers.sentence_parser import MySentenceSplitter
from pairag.file.models.file_item import FileItem
from pairag.file.readers.base import BaseReader
from pairag.file.readers.csv_reader import CSVReader
from pairag.file.readers.doc_reader import DocxReader
from pairag.file.readers.excel_reader import ExcelReader
from pairag.file.readers.faq_reader import FAQReader
from pairag.file.readers.html_reader import HtmlReader
from pairag.file.readers.image_reader import ImageReader
from pairag.file.readers.jsonl2md_reader import Json2MdReader
from pairag.file.readers.markdown_reader import MarkdownReader
from pairag.file.readers.mineru_reader import MineruPdfReader
from pairag.file.readers.pptx_reader import PptxReader
from pairag.file.readers.text_reader import TextReader
from pairag.file.readers.simple_pdf_reader import SimplePdfReader
from pairag.file.readers.video_reader import VideoReader
from pairag.file.nodeparsers.token_parser import TokenTextSplitter
from pairag.file.nodeparsers.paragraph_parser import ParagraphSplitter
from pairag.file.nodeparsers.pai_markdown_parser import MarkdownNodeParser
from pairag.file.nodeparsers.positional_markdown_parser import PositionalMarkdownNodeParser
from pairag.file.store.base import BaseFileStore
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.video_caption_tool import VideoCaptionTool
from pairag.file.utils.tokenization import estimate_tokens_in_text
from pairag.file.utils.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_PARAGRAPH_SEPARATOR,
    DEFAULT_PARSER_TYPE,
)
from pairag.file.utils.image_utils import MARKDOWN_IMAGE_PATTERN, markdown_image_text_to_chunk
from llama_index.core.bridge.pydantic import Field, BaseModel
from typing import Optional
import re
import os


IMAGE_DOC_TYPES = set([".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".heic", ".webp"])
DOC_TYPES_DO_NOT_NEED_CHUNKING = set([".csv", ".xlsx", ".xls", ".jsonl"])
DOC_TYPES_CONVERT_TO_MD = set([".md", ".pdf", ".docx", ".htm", ".html", ".pptx", ".doc", ".ppt"])
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


class TableParserConfig(BaseModel):
    """Configuration for table parser (CSV/Excel). Only used when parser_type == 'table'."""
    concat_rows: bool = Field(default=False, description="Whether to concatenate all rows into one document")
    row_joiner: str = Field(default="\n", description="Separator to use for joining each row")
    header_index_max: Optional[int] = Field(default=0, description="Maximum row index to use as header")
    format_sheet_data_to_json: bool = Field(default=False, description="Whether to format sheet data as JSON")
    sheet_column_filters: Optional[List[str]] = Field(default=None, description="List of column names to filter")
    question_column_index: Optional[int] = Field(default=0, description="Index of question column")
    answer_column_index: Optional[int] = Field(default=1, description="Index of answer column")



class ChunkConfig(BaseModel):
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP)
    parser_type: str = Field(default=DEFAULT_PARSER_TYPE)
    separator: str = Field(default=DEFAULT_PARAGRAPH_SEPARATOR)
    image_caption_model: Optional[str] = Field(default=None)
    image_caption_provider_name: str = Field(default="openai_like")
    table_config: Optional[TableParserConfig] = Field(default_factory=TableParserConfig, description="Table parser configuration (only used when parser_type == 'table')")


class ReaderConfig(BaseModel):
    enable_mineru: bool = Field(default=False)
    mineru_endpoint: str = Field(default="")
    mineru_token: str = Field(default="")


def node_id_func(i: int, doc: BaseNode) -> str:
    """Generate a short chunk ID: ``chunk_xxxxxxxx`` (8 hex chars).

    Uses 8 random hex chars (32 bits) for ~4 billion combinations — sufficient
    to avoid collisions in practice while keeping IDs short for model inference.
    """
    return f"chunk_{uuid.uuid4().hex[:8]}"



class FileParser:
    def __init__(
        self,
        file_store: BaseFileStore,
        image_caption_tool: ImageCaptionTool = None,
        video_caption_tool: VideoCaptionTool = None,
        reader_config: Optional[ReaderConfig] = ReaderConfig(),
        chunk_config: Optional[ChunkConfig] = ChunkConfig(),
    ):
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        self.video_caption_tool = video_caption_tool
        self.chunk_config = chunk_config
        self.reader_config = reader_config


    def _get_reader(
            self,
            file_extension: str,
            is_attachment: bool=False,
            chunk_config: Optional[ChunkConfig] = None) -> BaseReader:
        chunk_size = chunk_config.chunk_size if chunk_config else DEFAULT_CHUNK_SIZE
        parser_type = chunk_config.parser_type if chunk_config else DEFAULT_PARSER_TYPE
        
        table_config = chunk_config.table_config if chunk_config and chunk_config.table_config else TableParserConfig()
        
        if is_attachment:
            match file_extension:
                case ".doc":
                    return DocxReader(file_store=self.file_store)
                case ".ppt":
                    return PptxReader(file_store=self.file_store)
                case ".docx":
                    return DocxReader(file_store=self.file_store)
                case ".pdf":
                    return SimplePdfReader(extract_images=False, file_store=self.file_store)
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
                case ".heic":
                    return ImageReader(
                        file_store=self.file_store,
                    )
                case ".webp":
                    return ImageReader(
                        file_store=self.file_store,
                    )
                case ".svg":
                    return ImageReader(
                        file_store=self.file_store,
                    )
                case ".bmp":
                    return ImageReader(
                        file_store=self.file_store,
                    )
                case ".gif":
                    return ImageReader(
                        file_store=self.file_store,
                    )
                case ".xlsx":
                    return ExcelReader()
                case ".xls":
                    return ExcelReader()
                case ".pptx":
                    return PptxReader(
                        file_store=self.file_store,
                    )
                case ".csv":
                    return CSVReader()
                case ".jsonl":
                    return Json2MdReader(chunk_size=chunk_size)
                case ".mp4":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".avi":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".mov":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".wmv":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".flv":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".mkv":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
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
                case ".doc":
                    return DocxReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".ppt":
                    return PptxReader(
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
                    if self.reader_config.enable_mineru or os.environ.get("ENABLE_MINERU", "false").lower() == "true":
                        return MineruPdfReader(
                            endpoint=self.reader_config.mineru_endpoint,
                            token=self.reader_config.mineru_token,
                            file_store=self.file_store,
                            image_caption_tool=self.image_caption_tool,
                        )
                    else:
                        return SimplePdfReader(
                            extract_images=True,
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
                case ".heic":
                    return ImageReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".webp":
                    return ImageReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".svg":
                    return ImageReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".bmp":
                    return ImageReader(
                        file_store=self.file_store,
                        image_caption_tool=self.image_caption_tool,
                    )
                case ".gif":
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
                    if parser_type.lower() == "faq":
                        return FAQReader(
                            header_index_max=table_config.header_index_max,
                            question_column_index=table_config.question_column_index,
                            answer_column_index=table_config.answer_column_index,
                        )
                    return ExcelReader(
                            concat_rows=table_config.concat_rows,
                            row_joiner=table_config.row_joiner,
                            header_index_max=table_config.header_index_max,
                            format_sheet_data_to_json=table_config.format_sheet_data_to_json,
                            sheet_column_filters=table_config.sheet_column_filters,
                        )
                case ".xls":
                    if parser_type.lower() == "faq":
                        return FAQReader(
                            header_index_max=table_config.header_index_max,
                            question_column_index=table_config.question_column_index,
                            answer_column_index=table_config.answer_column_index,
                        )
                    return ExcelReader(
                            concat_rows=table_config.concat_rows,
                            row_joiner=table_config.row_joiner,
                            header_index_max=table_config.header_index_max,
                            format_sheet_data_to_json=table_config.format_sheet_data_to_json,
                            sheet_column_filters=table_config.sheet_column_filters,
                        )
                case ".csv":
                    return CSVReader(
                            concat_rows=table_config.concat_rows,
                            row_joiner=table_config.row_joiner,
                            header_index_max=table_config.header_index_max,
                            format_sheet_data_to_json=table_config.format_sheet_data_to_json,
                            sheet_column_filters=table_config.sheet_column_filters,
                        )
                case ".jsonl":
                    return Json2MdReader(chunk_size=chunk_size)
                case ".txt":
                    return TextReader()
                case ".mp4":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".avi":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".mov":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".wmv":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".flv":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
                case ".mkv":
                    return VideoReader(
                        file_store=self.file_store,
                        video_caption_tool=self.video_caption_tool,
                    )
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
            chunk_config=chunk_config)
        return reader.read(file_item)

    def split_docs(
        self, docs: List[Document], chunk_config: ChunkConfig
    ) -> List[BaseNode]:
        splitted_nodes: List[BaseNode] = []

        for doc_node in docs:
            chunks = []
            doc_type = doc_node.metadata["file_extension"]
            if doc_type in IMAGE_DOC_TYPES:
                node_id = node_id_func(0, doc_node)
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
                token_count = estimate_tokens_in_text(doc_node.text)

                if token_count > chunk_config.chunk_size:
                    parser = TokenTextSplitter(
                        chunk_size=chunk_config.chunk_size,
                        chunk_overlap=0,
                        id_func=node_id_func,
                    )
                    chunks = parser.get_nodes_from_documents([doc_node])
                else:
                    chunks.append(TextNode(
                        id_=node_id_func(0, doc_node),
                        text=doc_node.text,
                        metadata={
                            "token_count": token_count,
                        }
                    ))
            elif chunk_config.parser_type.lower() == "token":
                parser = TokenTextSplitter(
                    chunk_size=chunk_config.chunk_size,
                    chunk_overlap=chunk_config.chunk_overlap,
                    id_func=node_id_func,
                )
                chunks = parser.get_nodes_from_documents([doc_node])
            elif chunk_config.parser_type.lower() == "paragraph":
                parser = ParagraphSplitter(
                    chunk_size=chunk_config.chunk_size,
                    chunk_overlap=chunk_config.chunk_overlap,
                    paragraph_separator=chunk_config.separator,
                    id_func=node_id_func,
                )
                chunks = parser.get_nodes_from_documents([doc_node])
            else:
                logger.info(f"Start splitting document: {doc_node.metadata['file_name']} with id {doc_node.id_}")
                parser = MySentenceSplitter(
                    id_func=node_id_func,
                    chunk_size=chunk_config.chunk_size,
                    chunk_overlap=chunk_config.chunk_overlap,
                    paragraph_separator=chunk_config.separator,
                    include_metadata=False,
                )
                if doc_type == ".pdf":
                    # 需要计算bbox
                    positional_md_node_parser = PositionalMarkdownNodeParser(
                        chunk_size=chunk_config.chunk_size,
                        chunk_overlap=chunk_config.chunk_overlap,
                        id_func=node_id_func,
                    )
                    chunks = positional_md_node_parser.get_nodes_from_documents([doc_node])
                elif doc_type in DOC_TYPES_CONVERT_TO_MD:
                    # markdown格式(md, html, doc 等)
                    md_node_parser = MarkdownNodeParser(
                        chunk_size=chunk_config.chunk_size,
                        chunk_overlap=chunk_config.chunk_overlap,
                        paragraph_separator=chunk_config.separator,
                        base_parser=parser,
                        id_func=node_id_func,
                    )
                    chunks = md_node_parser.get_nodes_from_documents([doc_node])
                else:
                    # txt格式等纯文本
                    chunks = parser.get_nodes_from_documents([doc_node])
                logger.info(
                    f"Finished split document into {len(chunks)} chunks: {doc_node.metadata['file_name']}"
                )

            for chunk in chunks:
                chunk.metadata.update(doc_node.metadata)
                chunk.metadata["doc_id"] = doc_node.id_
                chunk.relationships = {
                            NodeRelationship.SOURCE: RelatedNodeInfo(
                                node_id=doc_node.id_, metadata={}
                            ),
                        }
                if "token_count" not in chunk.metadata:
                    chunk.metadata["token_count"] = estimate_tokens_in_text(chunk.text)
            splitted_nodes.extend(chunks)

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
