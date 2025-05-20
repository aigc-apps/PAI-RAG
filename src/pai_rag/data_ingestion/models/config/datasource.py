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
    pai_rag_token: Optional[str] = None
    pai_rag_endpoint: Optional[str] = None
    pai_rag_knowledgebase: Optional[str] = "default"
    pai_rag_embed_dims: Optional[int] = 1024
