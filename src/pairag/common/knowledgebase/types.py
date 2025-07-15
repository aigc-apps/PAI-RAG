from enum import Enum


class VectorIndexRetrievalType(str, Enum):
    vector = "vector"
    fulltext = "fulltext"
    hybrid = "hybrid"


class FileStatus(str, Enum):
    pending = "pending"  # file is uploaded but not processed
    parsing = "parsing"  # parsing file
    persisting = "persisting"  # file is persisting (including embedding)
    succeeded = "succeeded"  # file process succeeded is ready for searching
    failed = "failed"  # file failed


class ChunkStatus(str, Enum):
    pending = "pending"
    succeeded = "succeeded"
    failed = "failed"
