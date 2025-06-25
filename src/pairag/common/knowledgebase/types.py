from enum import Enum


class VectorIndexRetrievalType(str, Enum):
    vector = "vector"
    fulltext = "fulltext"
    hybrid = "hybrid"
