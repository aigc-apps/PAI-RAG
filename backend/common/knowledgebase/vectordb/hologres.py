from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection


class HologresConnection(BaseVectorDbConnection):
    """
    Hologres VectorDb Config
    """
    type: VectorDbType = VectorDbType.HOLOGRES

    host: str
    port: int = 80
    user: str
    encrypted_password: str
    database: str
