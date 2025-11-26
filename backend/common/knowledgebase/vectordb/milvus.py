from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection


class MilvusConnection(BaseVectorDbConnection):
    """
    Milvus VectorDb Config
    """
    type: VectorDbType = VectorDbType.MILVUS

    host: str
    port: int = 19530
    user: str
    encrypted_password: str
    database: str = "default"
