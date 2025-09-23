from typing import Dict, Type

from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection
from common.knowledgebase.vectordb.elastic import ElasticsearchConnection
from common.knowledgebase.vectordb.local import LocalConnection
from common.knowledgebase.vectordb.milvus import MilvusConnection
from common.knowledgebase.vectordb.postgres import PostgresqlConnection
from config.utils.vectordb import create_vector_db_connection_from_env
from db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import SQLModel
from db.models.vectordb import VectorDbConfig
from config.providers.base_provider import BaseConfigProvider
from pydantic import Field


DEFAULT_VECTOR_ID = "default_vectordb"


@with_async_db_session
async def get_vector_db_connection_from_db(
    session: AsyncSession,
) -> BaseVectorDbConnection:
    """
    Get the vector db connection from the database.
    """
    vector_db_config = await session.get(VectorDbConfig, DEFAULT_VECTOR_ID)
    if vector_db_config is None:
        return create_vector_db_connection_from_env()

    return create_vector_db_connection_from_dict(vector_db_config.config)


def create_vector_db_connection_from_dict(config: dict) -> BaseVectorDbConnection:
    vector_db_type = config.get("type", "local")
    if vector_db_type == VectorDbType.LOCAL:
        return LocalConnection()
    elif vector_db_type == VectorDbType.ELASTICSEARCH:
        return ElasticsearchConnection.from_dict(config)
    elif vector_db_type == VectorDbType.MILVUS:
        return MilvusConnection.from_dict(config)
    elif vector_db_type == VectorDbType.POSTGRESQL:
        return PostgresqlConnection.from_dict(config)
    else:
        raise ValueError(f"Invalid vector db type: {vector_db_type}")


class VectorDbProvider(BaseConfigProvider):
    config_map: Dict[str, VectorDbConfig] = Field(default={})
    entity_class: Type[SQLModel] = VectorDbConfig

    def get_vector_db_connection(self):
        vector_db_config = self.config_map.get(DEFAULT_VECTOR_ID)
        if vector_db_config is None:
            return create_vector_db_connection_from_env()

        return create_vector_db_connection_from_dict(vector_db_config.config)


vectordb_provider = VectorDbProvider()
