from abc import ABC, abstractmethod
import os
from typing import List
from loguru import logger

from llama_index.core.schema import TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding

from pai_rag.integrations.data_analysis.text2sql.utils.constants import (
    DESCRIPTION_STORAGE_PATH,
    HISTORY_STORAGE_PATH,
    VALUE_STORAGE_PATH,
)
from pai_rag.integrations.index.pai.vector_store_config import FaissVectorStoreConfig
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex


class DBInfoIndex(ABC):
    """
    index接口，内部使用
    """

    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        self._db_name = db_name
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k

    @abstractmethod
    def insert_nodes(self, nodes: List[TextNode]):
        pass

    @abstractmethod
    def as_retriever(self):
        pass


class SchemaIndex(DBInfoIndex):
    def __init__(self, db_name, embed_model, similarity_top_k):
        super().__init__(db_name, embed_model, similarity_top_k)
        description_persist_path = os.path.join(DESCRIPTION_STORAGE_PATH, db_name)
        description_vector_store_config = FaissVectorStoreConfig(
            persist_path=description_persist_path
        )
        self._description_index = PaiVectorStoreIndex(
            vector_store_config=description_vector_store_config,
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
        )

    def insert_nodes(self, nodes: List[TextNode]):
        logger.info(f"start inserting schema for {self._db_name}")
        return self._description_index.insert_nodes(nodes)

    def as_retriever(self):
        return self._description_index.as_retriever()


class HistoryIndex(DBInfoIndex):
    def __init__(self, db_name, embed_model, similarity_top_k):
        super().__init__(db_name, embed_model, similarity_top_k)
        history_persist_path = os.path.join(HISTORY_STORAGE_PATH, db_name)
        history_vector_store_config = FaissVectorStoreConfig(
            persist_path=history_persist_path,
        )
        self._history_index = PaiVectorStoreIndex(
            embed_model=embed_model,
            vector_store_config=history_vector_store_config,
            similarity_top_k=similarity_top_k,
        )

    def insert_nodes(self, nodes: List[TextNode]):
        logger.info("start inserting history...")
        return self._history_index.insert_nodes(nodes)

    def as_retriever(self):
        return self._history_index.as_retriever()


class ValueIndex(DBInfoIndex):
    def __init__(self, db_name, embed_model, similarity_top_k):
        super().__init__(db_name, embed_model, similarity_top_k)
        value_persist_path = os.path.join(VALUE_STORAGE_PATH, db_name)
        value_vector_store_config = FaissVectorStoreConfig(
            persist_path=value_persist_path,
        )
        self._value_index = PaiVectorStoreIndex(
            vector_store_config=value_vector_store_config,
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
        )

    def insert_nodes(self, nodes: List[TextNode]):
        logger.info(f"start inserting value for {self._db_name}")
        return self._value_index.insert_nodes(nodes)

    def as_retriever(self):
        return self._value_index.as_retriever()
