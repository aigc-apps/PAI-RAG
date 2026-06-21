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


class DataSourceType(str, Enum):
    llms_txt = "llms_txt"  # sites exposing an official llms.txt manifest (e.g. help.aliyun.com)
    sphinx = "sphinx"  # readthedocs / Sphinx sites
    local = "local"  # user-uploaded local files
    github = "github"  # github repo docs


class DataSourceStatus(str, Enum):
    """Aggregate state of a data source (two-phase, see plan §1.4)."""
    idle = "idle"  # never synced or no run in progress
    syncing = "syncing"  # phase A: discover/diff/fetch/enqueue in progress
    ingesting = "ingesting"  # phase A done, files enqueued and parsing async
    succeeded = "succeeded"  # all documents synced
    partial = "partial"  # some documents failed to parse
    failed = "failed"  # phase A itself failed
    cancelled = "cancelled"  # sync cancelled by the user


class DataSourceDocStatus(str, Enum):
    """Per-document sync state within a data source."""
    discovered = "discovered"  # listed by adapter, not yet fetched
    fetching = "fetching"  # body being fetched
    ingesting = "ingesting"  # enqueued into the KB ingestion pipeline
    synced = "synced"  # parsed/embedded successfully
    failed = "failed"  # fetch or parse failed
    cancelled = "cancelled"  # parse cancelled by the user
    deleted = "deleted"  # removed from source, cleaned up


class SyncRunStatus(str, Enum):
    running = "running"
    succeeded = "succeeded"
    partial = "partial"
    failed = "failed"


class SyncTrigger(str, Enum):
    manual = "manual"
    scheduled = "scheduled"


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
