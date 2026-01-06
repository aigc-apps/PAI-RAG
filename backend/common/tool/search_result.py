from pydantic import BaseModel
from typing import List


class SearchResult(BaseModel):
    id: str | None = None
    title: str | None = None
    content: str | None = None
    url: str | None = None
    score: float = 0
    favicon: str | None = None
    hostname: str | None = None
    publish_time: str | None = None
    images: List[dict] | None = []
    metadata: dict | None = None
