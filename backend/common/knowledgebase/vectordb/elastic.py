from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection


class ElasticsearchConnection(BaseVectorDbConnection):
    """
    Elasticsearch connection
    """
    type: VectorDbType = VectorDbType.ELASTICSEARCH
    endpoint: str
    user: str
    encrypted_password: str
