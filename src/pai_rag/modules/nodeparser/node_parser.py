import hashlib
from typing import Any, Dict, List
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import BaseNode
from llama_index.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from pai_rag.utils.constants import (
    DEFAULT_PARAGRAPH_SEP,
    SENTENCE_CHUNK_OVERLAP,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_BREAKPOINT,
    DEFAULT_BUFFER_SIZE,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG


def node_id_hash(i: int, doc: BaseNode) -> str:
    encoded_raw_text = f"""<<{i}>>{doc.metadata["file_name"]}""".encode()
    hash = hashlib.sha256(encoded_raw_text).hexdigest()
    return hash


class NodeParserModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["EmbeddingModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.embed_model = new_params["EmbeddingModule"]
        print("loading node parsers")
        parser_config = new_params[MODULE_PARAM_CONFIG]
        if parser_config["type"] == "Token":
            return TokenTextSplitter(
                chunk_size=parser_config.get("chunk_size", DEFAULT_CHUNK_SIZE),
                chunk_overlap=parser_config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
                id_func=node_id_hash,
            )
        elif parser_config["type"] == "Sentence":
            return SentenceSplitter(
                chunk_size=parser_config.get("chunk_size", DEFAULT_CHUNK_SIZE),
                chunk_overlap=parser_config.get(
                    "chunk_overlap", SENTENCE_CHUNK_OVERLAP
                ),
                paragraph_separator=parser_config.get(
                    "paragraph_separator", DEFAULT_PARAGRAPH_SEP
                ),
                id_func=node_id_hash,
            )
        elif parser_config["type"] == "SentenceWindow":
            return SentenceWindowNodeParser(
                sentence_splitter=SentenceSplitter(
                    chunk_size=parser_config.get("chunk_size", DEFAULT_CHUNK_SIZE),
                    chunk_overlap=parser_config.get(
                        "chunk_overlap", SENTENCE_CHUNK_OVERLAP
                    ),
                    paragraph_separator=parser_config.get(
                        "paragraph_separator", DEFAULT_PARAGRAPH_SEP
                    ),
                    id_func=node_id_hash,
                ).split_text,
                window_size=parser_config.get("window_size", DEFAULT_WINDOW_SIZE),
            )
        elif parser_config["type"] == "Semantic":
            return SemanticSplitterNodeParser(
                embed_model=self.embed_model,
                breakpoint_percentile_threshold=parser_config.get(
                    "breakpoint_percentile_threshold", DEFAULT_BREAKPOINT
                ),
                buffer_size=parser_config.get("buffer_size", DEFAULT_BUFFER_SIZE),
            )
        else:
            raise ValueError(f"Unknown Splitter Type: {parser_config['type']}")
