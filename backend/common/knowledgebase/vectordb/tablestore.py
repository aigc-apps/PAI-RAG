from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection


class TablestoreConnection(BaseVectorDbConnection):
    """
    Tablestore VectorDb Config
    """
    type: VectorDbType = VectorDbType.TABLESTORE

    endpoint: str
    instance_name: str
    access_key_id: str
    access_key_secret: str
