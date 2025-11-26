from enum import Enum
from rag.vector_store.local import LocalChromaVectorStore
from llama_index.vector_stores.alibabacloud_opensearch import AlibabaCloudOpenSearchStore
from llama_index.vector_stores.hologres import HologresVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore


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
    cancelled = "cancelled" # file process cancelled


class ChunkStatus(str, Enum):
    pending = "pending"
    succeeded = "succeeded"
    failed = "failed"


class VectorDbType(str, Enum):
    OPENSEARCH = "opensearch"
    ELASTICSEARCH = "elasticsearch"
    ANALYTICDB = "analyticdb"
    POSTGRESQL = "postgresql"
    HOLOGRES = "hologres"
    TABLESTORE = "tablestore"
    MILVUS = "milvus"
    DASHVECTOR = "dashvector"
    LOCAL = "local"



SUPPORTED_VECTOR_DB_TYPES = [
    "local",
    "milvus",
    "postgresql",
    "elasticsearch",
    "hologres",
    "opensearch",
    "tablestore",
]

# 不支持全文检索和混合检索的向量数据库类型列表
FULLTEXT_UNSUPPORTED_VECTOR_STORE_TYPES = (
    LocalChromaVectorStore,
    AlibabaCloudOpenSearchStore,
    HologresVectorStore,
)

def is_fulltext_supported_by_vector_store(vector_store: BasePydanticVectorStore) -> bool:
    """Determine whether the active vector store can execute fulltext/hybrid queries."""
    return not isinstance(vector_store, FULLTEXT_UNSUPPORTED_VECTOR_STORE_TYPES)
