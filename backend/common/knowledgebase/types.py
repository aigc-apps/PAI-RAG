from enum import Enum


class VectorIndexRetrievalType(str, Enum):
    vector = "vector"
    fulltext = "fulltext"
    hybrid = "hybrid"


def is_fulltext_supported(vector_db_type: str) -> bool:
    """
    Check if the vector database type supports fulltext and hybrid retrieval.

    Args:
        vector_db_type: The vector database type string

    Returns:
        True if fulltext/hybrid retrieval is supported, False otherwise
    """
    from common.knowledgebase.types import VectorDbType, VECTOR_DB_TYPES_WITHOUT_FULLTEXT
    try:
        db_type_enum = VectorDbType(vector_db_type.lower())
        return db_type_enum not in VECTOR_DB_TYPES_WITHOUT_FULLTEXT
    except ValueError:
        # 如果类型不在枚举中，默认支持（向后兼容）
        return True


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
VECTOR_DB_TYPES_WITHOUT_FULLTEXT = [
    VectorDbType.LOCAL,
    VectorDbType.OPENSEARCH,
    VectorDbType.HOLOGRES,
]
