from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection


class PostgresqlConnection(BaseVectorDbConnection):
    """
    Postgresql VectorDb Config
    """
    type: VectorDbType = VectorDbType.POSTGRESQL

    host: str
    port: int = 5432
    user: str
    encrypted_password: str
    database: str
