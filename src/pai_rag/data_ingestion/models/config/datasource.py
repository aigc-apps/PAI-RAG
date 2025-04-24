from typing import List, Optional
from pydantic import BaseModel


class DataSourceConfig(BaseModel):
    input_path: str
    output_path: str
    enable_delta: bool = False
    file_extensions: Optional[List[str]] = None