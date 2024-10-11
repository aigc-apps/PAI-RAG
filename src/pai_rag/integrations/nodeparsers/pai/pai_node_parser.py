import hashlib
import os
import re
from typing import List, Any, Dict
from llama_index.core.schema import BaseNode, TextNode, ImageDocument, ImageNode
from llama_index.core.schema import TransformComponent
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from pydantic import BaseModel
from pai_rag.integrations.nodeparsers.base import MarkdownNodeParser
from pai_rag.utils.constants import (
    DEFAULT_NODE_PARSER_TYPE,
    DEFAULT_PARAGRAPH_SEP,
    DEFAULT_SENTENCE_CHUNK_OVERLAP,
    DEFAULT_SENTENCE_WINDOW_SIZE,
    DEFAULT_BREAKPOINT,
    DEFAULT_BUFFER_SIZE,
)
import logging

logger = logging.getLogger(__name__)


class NodeParserConfig(BaseModel):
    type: str = DEFAULT_NODE_PARSER_TYPE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    enable_multimodal: bool = False
    paragraph_separator: str = DEFAULT_PARAGRAPH_SEP
    sentence_window_size: int = DEFAULT_SENTENCE_WINDOW_SIZE
    sentence_chunk_overlap: int = DEFAULT_SENTENCE_CHUNK_OVERLAP
    breakpoint_percentile_threshold: float = DEFAULT_BREAKPOINT
    buffer_size: int = DEFAULT_BUFFER_SIZE


DOC_TYPES_DO_NOT_NEED_CHUNKING = set(
    [".csv", ".xlsx", ".xls", ".htm", ".html", ".jsonl"]
)
IMAGE_FILE_TYPES = set([".jpg", ".jpeg", ".png"])

IMAGE_URL_REGEX = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|jpeg|png)",
    re.IGNORECASE,
)

COMMON_FILE_PATH_FODER_NAME = "__pairag__knowledgebase__"


def format_temp_file_path(temp_file_path):
    path_components = temp_file_path.split(f"{COMMON_FILE_PATH_FODER_NAME}/")
    return path_components[-1]


def node_id_hash(i: int, doc: BaseNode) -> str:
    encoded_raw_text = (
        f"""<<{i}>>{doc.metadata.get("file_name", "DUMMY_FILE_NAME")}""".encode()
    )
    hash = hashlib.sha256(encoded_raw_text).hexdigest()
    return hash


def get_data_parser(parser_config: NodeParserConfig):
    parser_type = parser_config.type.lower()
    if parser_type == "token":
        return TokenTextSplitter(
            chunk_size=parser_config.chunk_size,
            chunk_overlap=parser_config.chunk_overlap,
            id_func=node_id_hash,
        )
    elif parser_type == "sentence":
        return SentenceSplitter(
            chunk_size=parser_config.chunk_size,
            chunk_overlap=parser_config.sentence_chunk_overlap,
            paragraph_separator=parser_config.paragraph_separator,
            id_func=node_id_hash,
        )
    elif parser_type == "sentencewindow":
        return SentenceWindowNodeParser(
            sentence_splitter=SentenceSplitter(
                chunk_size=parser_config.chunk_size,
                chunk_overlap=parser_config.sentence_chunk_overlap,
                paragraph_separator=parser_config.paragraph_separator,
                id_func=node_id_hash,
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
    _parser_config: Any = PrivateAttr()
    _parser: Any = PrivateAttr()
    _doc_cnt_map: Any = PrivateAttr()

    def __init__(self, parser_config: NodeParserConfig = None):
        super().__init__()
        self._parser_config = parser_config or NodeParserConfig()
        self._parser = get_data_parser(self._parser_config)
        self._doc_cnt_map = {}

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
        # Accumulate node index for doc
        splitted_nodes = []
        self._doc_cnt_map = {}

        for doc_node in nodes:
            doc_node.metadata["file_path"] = format_temp_file_path(
                doc_node.metadata["file_path"]
            )
            doc_type = self._extract_file_type(doc_node.metadata)
            doc_key = f"""{doc_node.metadata.get("file_path", "dummy")}"""
            if isinstance(doc_node, ImageDocument):
                node_id = node_id_hash(
                    self._get_auto_increment_node_id(doc_key), doc_node
                )
                splitted_nodes.append(
                    ImageNode(
                        id_=node_id,
                        text=doc_node.text,
                        metadata=doc_node.metadata,
                        image_url=doc_node.image_url,
                        image_path=doc_node.image_path,
                        image_mimetype=doc_node.image_mimetype,
                    )
                )
            elif doc_type in DOC_TYPES_DO_NOT_NEED_CHUNKING:
                node_id = node_id_hash(
                    self._get_auto_increment_node_id(doc_key), doc_node
                )
                splitted_nodes.append(
                    TextNode(
                        id_=node_id, text=doc_node.text, metadata=doc_node.metadata
                    )
                )
            else:
                if doc_type == ".md" or doc_type == ".pdf":
                    md_node_parser = MarkdownNodeParser(
                        id_func=node_id_hash,
                        enable_multimodal=self._parser_config.enable_multimodal,
                        base_parser=self._parser,
                    )
                    tmp_nodes = md_node_parser.get_nodes_from_documents([doc_node])
                else:
                    tmp_nodes = self._parser.get_nodes_from_documents([doc_node])
                if tmp_nodes:
                    for tmp_node in tmp_nodes:
                        tmp_node.id_ = node_id_hash(
                            self._get_auto_increment_node_id(doc_key), doc_node
                        )
                        splitted_nodes.append(tmp_node)

        for node in nodes:
            node.excluded_embed_metadata_keys.append("file_path")
            node.excluded_embed_metadata_keys.append("image_url")
            node.excluded_embed_metadata_keys.append("total_pages")
            node.excluded_embed_metadata_keys.append("source")

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
