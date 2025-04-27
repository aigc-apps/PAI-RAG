from typing import List
from pydantic import BaseModel


class DocItem(BaseModel):
    doc_path: str
    node_ids: List[str]
    modified_time: str
