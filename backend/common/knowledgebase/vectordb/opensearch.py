from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection


class OpensearchConnection(BaseVectorDbConnection):
    """
    Opensearch VectorDb Config
    """
    type: VectorDbType = VectorDbType.OPENSEARCH

    endpoint: str
    instance_id: str
    username: str
    encrypted_password: str
