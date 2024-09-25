from typing import Annotated, List, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum

DEFAULT_LOCAL_STORAGE_PATH = "./localdata/storage"


class SupportedVectorStoreType(str, Enum):
    faiss = "faiss"
    analyticdb = "analyticdb"
    elasticsearch = "elasticsearch"
    postgresql = "postgresql"
    opensearch = "opensearch"
    milvus = "milvus"
    hologres = "hologres"


class VectorIndexRetrievalType(str, Enum):
    keyword = "keyword"
    embedding = "embedding"
    hybrid = "hybrid"


VECTOR_STORE_TYPES_WITH_HYBRID_SEARCH = [
    SupportedVectorStoreType.elasticsearch,
    SupportedVectorStoreType.postgresql,
    SupportedVectorStoreType.milvus,
]


class BaseVectorStoreConfig(BaseModel):
    persist_path: str = None
    type: SupportedVectorStoreType
    is_image_store: bool = False

    @classmethod
    def get_subclasses(cls):
        return tuple(cls.__subclasses__())


class FaissVectorStoreConfig(BaseVectorStoreConfig):
    type: Literal[SupportedVectorStoreType.faiss] = SupportedVectorStoreType.faiss


class AnalyticDBVectorStoreConfig(BaseVectorStoreConfig):
    type: Literal[
        SupportedVectorStoreType.analyticdb
    ] = SupportedVectorStoreType.analyticdb
    ak: str
    sk: str
    region_id: str
    instance_id: str
    account: str
    account_password: str
    namespace: str
    collection: str


class HologresVectorStoreConfig(BaseVectorStoreConfig):
    type: Literal[SupportedVectorStoreType.hologres] = SupportedVectorStoreType.hologres
    host: str
    port: int
    user: str
    password: str
    database: str
    table_name: str
    pre_delete_table: bool = False


class ElasticSearchVectorStoreConfig(BaseVectorStoreConfig):
    type: Literal[
        SupportedVectorStoreType.elasticsearch
    ] = SupportedVectorStoreType.elasticsearch
    es_url: str
    es_user: str
    es_password: str
    es_index: str


class MilvusVectorStoreConfig(BaseVectorStoreConfig):
    type: Literal[SupportedVectorStoreType.milvus] = SupportedVectorStoreType.milvus
    host: str
    port: int
    user: str
    password: str
    database: str
    collection_name: str
    reranker_weights: List[float] = [0.5, 0.5]


class OpenSearchVectorStoreConfig(BaseVectorStoreConfig):
    type: Literal[
        SupportedVectorStoreType.opensearch
    ] = SupportedVectorStoreType.opensearch
    endpoint: str
    instance_id: str
    username: str
    password: str
    table_name: str


class PostgreSQLVectorStoreConfig(BaseVectorStoreConfig):
    type: Literal[
        SupportedVectorStoreType.postgresql
    ] = SupportedVectorStoreType.postgresql
    host: str
    port: int
    database: str
    table_name: str = "default"
    username: str
    password: str


class PaiVectorIndexConfig(BaseModel):
    vector_store: Annotated[
        Union[BaseVectorStoreConfig.get_subclasses()], Field(discriminator="type")
    ]
    enable_multimodal: bool = False
    persist_path: str = DEFAULT_LOCAL_STORAGE_PATH

    class Config:
        frozen = True


PaiVectorStoreConfig = Annotated[
    Union[BaseVectorStoreConfig.get_subclasses()], Field(discriminator="type")
]

if __name__ == "__main__":
    json_data = {
        "vector_store": {
            "persist_path": "./1234",
            "type": "Postgresql",
            "host": "abc.com",
            "port": 123,
            "database": "test",
            "table_name": "table",
            "username": "jim",
            "password": "123xx",
        }
    }
    config = PaiVectorIndexConfig.model_validate(json_data)
    print(config)

    VectorStoreConfig = Annotated[
        Union[BaseVectorStoreConfig.get_subclasses()],
        Field(discriminator="vectordb_type"),
    ]

    json_data = {
        "persist_path": "./123",
        "type": "postgresql",
        "host": "abc.com",
        "port": 123,
        "database": "test",
        "table_name": "table",
        "username": "jim",
        "password": "123xx",
    }
    config2 = BaseVectorStoreConfig.model_validate(json_data)
    print(config2)
