from typing import List, Optional
from pydantic import BaseModel


class DataSourceConfig(BaseModel):
    input_path: str
    output_path: str
    enable_delta: bool = False
    file_extensions: Optional[List[str]] = None

    # use langstudio index manifest
    target_index: Optional[str] = None
    target_index_version: Optional[str] = None

    # connect to rag service
    rag_api_key: Optional[str] = None
    rag_endpoint: Optional[str] = None
    knowledgebase: Optional[str] = "default"
    embed_dims: Optional[int] = 1024
