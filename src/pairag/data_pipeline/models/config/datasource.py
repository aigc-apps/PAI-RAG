from typing import List, Optional
from pydantic import BaseModel


class DataSourceConfig(BaseModel):
    input_path: str
    output_path: str
    enable_delta: bool = False
    file_extensions: Optional[List[str]] = None

    # connect to rag service
    pairag_token: Optional[str] = None
    pairag_endpoint: Optional[str] = None
    pairag_knowledgebase: Optional[str] = "default"
    pairag_embed_dims: Optional[int] = 1024
