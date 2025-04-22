from enum import Enum
from typing import List
from pydantic import BaseModel
from llama_index.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

from pai_rag.utils.constants import DEFAULT_NODE_PARSER_TYPE



class OperatorName(str, Enum):
    PARSER = "parser"
    SPLITTER = "splitter"
    EMBEDDER = "embedder"
    WRITER = "writer"


class BaseOperatorConfig(BaseModel):
    """
    Base class for operator configs.
    """
    name: OperatorName
    num_cpus: float = 1
    num_gpus: float = 0
    memory: float = 2

    input_path: str
    output_path: str
    model_dir: str = None


class ParserConfig(BaseOperatorConfig):
    """
    Config for parse operator.
    """
    name: OperatorName = OperatorName.PARSER
    enable_pdf_ocr: bool = False
    concat_sheet_rows: bool = False
    recursive: bool = True
    supported_file_extensions: List[str] = []


class SplitterConfig(BaseOperatorConfig):
    """
    Config for split operator.
    """
    name: OperatorName = OperatorName.SPLITTER
    paragraph_separator: str = None
    node_parser_type: str = DEFAULT_NODE_PARSER_TYPE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


class EmbedderConfig(BaseOperatorConfig):
    """
    Config for embed operator.
    """
    name: OperatorName = OperatorName.EMBEDDER
    model: str = "bge-m3"
    connection_name: str = None
    workspace_id: str = None
    enable_sparse: bool = False
    source: str = "huggingface"


