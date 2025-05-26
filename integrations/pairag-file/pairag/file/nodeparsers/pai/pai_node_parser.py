import os
import re
from typing import List, Any, Dict, Optional
import uuid
from llama_index.core.schema import BaseNode, TextNode, ImageDocument
from llama_index.core.schema import TransformComponent
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from llama_index.core import Settings
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from pydantic import BaseModel
from pairag.file.nodeparsers.pai.pai_markdown_parser import MarkdownNodeParser
from pairag.file.nodeparsers.pai.constants import (
    DEFAULT_NODE_PARSER_TYPE,
    DEFAULT_PARAGRAPH_SEP,
    DEFAULT_SENTENCE_CHUNK_OVERLAP,
    DEFAULT_SENTENCE_WINDOW_SIZE,
    DEFAULT_BREAKPOINT,
    DEFAULT_BUFFER_SIZE,
)
from loguru import logger

from pairag.file.nodeparsers.pai.image_caption_tool import ImageCaptionTool
from enum import Enum


class NodeParserType(str, Enum):
    TOKEN = "token"
    SENTENCE = "sentence"
    SENTENCE_WINDOW = "sentencewindow"
    SEMANTIC = "semantic"


class NodeParserConfig(BaseModel):
    type: Optional[str] = DEFAULT_NODE_PARSER_TYPE
    chunk_size: Optional[int] = DEFAULT_CHUNK_SIZE
    chunk_overlap: Optional[int] = DEFAULT_CHUNK_OVERLAP
    enable_multimodal: Optional[bool] = False
    paragraph_separator: Optional[str] = DEFAULT_PARAGRAPH_SEP
    sentence_window_size: Optional[int] = DEFAULT_SENTENCE_WINDOW_SIZE
    sentence_chunk_overlap: Optional[int] = DEFAULT_SENTENCE_CHUNK_OVERLAP
    breakpoint_percentile_threshold: Optional[float] = DEFAULT_BREAKPOINT
    buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE


DOC_TYPES_DO_NOT_NEED_CHUNKING = set([".csv", ".xlsx", ".xls", ".jsonl"])
DOC_TYPES_CONVERT_TO_MD = set([".md", ".pdf", ".docx", ".htm", ".html", ".pptx"])
IMAGE_FILE_TYPES = set([".jpg", ".jpeg", ".png"])

IMAGE_URL_REGEX = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|jpeg|png)",
    re.IGNORECASE,
)

COMMON_FILE_PATH_FODER_NAME = "__pairag__knowledgebase__"
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


def rand_node_id_hash(i: int, doc: BaseNode) -> str:
    return str(uuid.uuid4())


def get_data_parser(parser_config: NodeParserConfig) -> NodeParser:
    parser_type = parser_config.type.lower()
    if parser_type == "token":
        return TokenTextSplitter(
            chunk_size=parser_config.chunk_size,
            chunk_overlap=parser_config.chunk_overlap,
            id_func=rand_node_id_hash,
        )
    elif parser_type == "sentence":
        return SentenceSplitter(
            chunk_size=parser_config.chunk_size,
            chunk_overlap=parser_config.chunk_overlap,
            paragraph_separator=parser_config.paragraph_separator,
            id_func=rand_node_id_hash,
        )
    elif parser_type == "sentencewindow":
        return SentenceWindowNodeParser(
            sentence_splitter=SentenceSplitter(
                chunk_size=parser_config.chunk_size,
                chunk_overlap=parser_config.chunk_overlap,
                paragraph_separator=parser_config.paragraph_separator,
                id_func=rand_node_id_hash,
            ).split_text,
            window_size=parser_config.sentence_window_size,
        )
    elif parser_type == "semantic":
        return SemanticSplitterNodeParser(
            embed_model=Settings.embed_model,
            breakpoint_percentile_threshold=parser_config.breakpoint_percentile_threshold,
            buffer_size=parser_config.buffer_size,
        )
    else:
        raise ValueError(f"Unknown Splitter Type: {parser_config['type']}")


class PaiNodeParser(TransformComponent):
    _parser_config: NodeParserConfig = PrivateAttr()
    _image_caption_tool: ImageCaptionTool = PrivateAttr()
    _parser: NodeParser = PrivateAttr()
    _doc_cnt_map: Any = PrivateAttr()

    def __init__(
        self,
        parser_config: NodeParserConfig = None,
        caption_tool: ImageCaptionTool = None,
    ):
        super().__init__()
        self._parser_config = parser_config or NodeParserConfig()
        self._image_caption_tool = caption_tool
        self._parser = get_data_parser(self._parser_config)
        self._doc_cnt_map = {}

        self._caption_tool = caption_tool

    def _extract_image_info(self, image_path):
        assert (
            self._caption_tool is not None
        ), "Multimodal LLM must be provided for image processing."
        return self._caption_tool.extract_path(image_path)

    def _extract_file_type(self, metadata: Dict[str, Any]):
        file_name = metadata.get("file_name", "dummy.txt")
        return os.path.splitext(file_name)[1]

    def _get_auto_increment_node_id(self, doc_key):
        if doc_key not in self._doc_cnt_map:
            self._doc_cnt_map[doc_key] = 0
        start_id = self._doc_cnt_map[doc_key]
        self._doc_cnt_map[doc_key] += 1
        return start_id

    def get_nodes_from_documents(
        self, nodes: List[BaseNode], **kwargs: Any
    ) -> List[BaseNode]:
        splitted_nodes: List[BaseNode] = []

        for doc_node in nodes:
            logger.info(f"Start splitting document: {doc_node.metadata['file_name']}")
            doc_type = self._extract_file_type(doc_node.metadata)

            chunks = []
            if isinstance(doc_node, ImageDocument):
                # 图片格式文档
                # 图片仅有一张，直接使用doc_id作为node_id
                node_id = doc_node.doc_id
                image_text = self._extract_image_info(doc_node.metadata["file_path"])
                metadata = doc_node.metadata
                metadata["image_url"] = doc_node.image_url
                chunks.append(
                    TextNode(
                        id_=node_id,
                        text=image_text,
                        metadata=metadata,
                        relationships={
                            NodeRelationship.SOURCE: RelatedNodeInfo(
                                node_id=doc_node.node_id, metadata={}
                            ),
                        },
                    )
                )
            elif doc_type in DOC_TYPES_DO_NOT_NEED_CHUNKING:
                # 表格格式文档
                metadata = doc_node.metadata

                node_id = str(uuid.uuid4())
                chunks.append(
                    TextNode(
                        id_=node_id,
                        text=doc_node.text,
                        metadata=metadata,
                        relationships={
                            NodeRelationship.SOURCE: RelatedNodeInfo(
                                node_id=doc_node.node_id, metadata={}
                            ),
                        },
                    )
                )
            else:
                if doc_type in DOC_TYPES_CONVERT_TO_MD:
                    # markdown格式(pdf, md, html, doc 等)
                    md_node_parser = MarkdownNodeParser(
                        id_func=rand_node_id_hash,
                        image_caption_tool=self._image_caption_tool,
                        max_chunk_size=self._parser_config.chunk_size,
                        chunk_overlap_size=self._parser_config.chunk_overlap,
                        base_parser=self._parser,
                    )
                    chunks = md_node_parser.get_nodes_from_documents([doc_node])
                else:
                    # txt格式等纯文本
                    chunks = self._parser.get_nodes_from_documents([doc_node])

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
            f"[DataReader] Split {len(nodes)} documents into {len(splitted_nodes)} nodes."
        )

        return splitted_nodes

    async def aget_nodes_from_documents(
        self, nodes: List[BaseNode], **kwargs: Any
    ) -> List[BaseNode]:
        return self.get_nodes_from_documents(nodes=nodes, **kwargs)

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        return self.get_nodes_from_documents(nodes, **kwargs)

    async def acall(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        return await self.aget_nodes_from_documents(nodes, **kwargs)
