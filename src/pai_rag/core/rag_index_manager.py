import os
import threading
from typing import Annotated, Union, Dict
from pydantic import BaseModel, Field
from pai_rag.core.rag_config import RagConfig
from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
)
from pai_rag.integrations.index.pai.vector_store_config import BaseVectorStoreConfig
import logging

logger = logging.getLogger(__name__)

DEFAULT_INDEX_FILE = "localdata/default__rag__index.json"
DEFAULT_INDEX_NAME = "default_index"
DEFAULT_MAX_INDEX_ENTRY_COUNT = os.environ.get("DEFAULT_MAX_INDEX_ENTRY_COUNT", 20)


"""
IndexEntry Model
An index entry should consist of name, embedding and vector_store settings.
"""


class RagIndexEntry(BaseModel):
    index_name: str = Field(
        default=DEFAULT_INDEX_NAME,
        description="Index name.",
        pattern=r"^[0-9a-zA-Z_-]{3, 20}$",
    )
    vector_store_config: Annotated[
        Union[BaseVectorStoreConfig.get_subclasses()], Field(discriminator="type")
    ]
    embedding_config: Annotated[
        Union[PaiBaseEmbeddingConfig.get_subclasses()], Field(discriminator="source")
    ]


"""
IndexMap Model.
Holds all index entries.
"""


class RagIndexMap(BaseModel):
    indexes: Dict[str, RagIndexEntry] = {}
    current_index_name: str = DEFAULT_INDEX_NAME


"""
Manages the index map.
"""


class RagIndexManager:
    def __init__(
        self,
        index_file: str,
        index_map: RagIndexMap,
    ):
        self._index_file = index_file
        self._index_map = index_map
        self._lock = threading.Lock()

    def add_default_index(self, rag_config: RagConfig):
        if DEFAULT_INDEX_NAME not in self._index_map.indexes:
            self._index_map.indexes[DEFAULT_INDEX_NAME] = RagIndexEntry(
                index_name=DEFAULT_INDEX_NAME,
                vector_store_config=rag_config.index.vector_store,
                embedding_config=rag_config.embedding,
            )

    @classmethod
    def from_file(cls, index_file: str):
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index_json_str = f.read()
                index_map = RagIndexMap.model_validate_json(index_json_str)
        else:
            index_map = RagIndexMap()

        return cls(index_file=index_file, index_map=index_map)

    def get_index_map(self) -> RagIndexMap:
        return self._index_map

    def get_index_by_name(self, index_name) -> RagIndexEntry:
        if not index_name:
            return self._index_map.indexes[self._index_map.current_index_name]
        return self._index_map.indexes[index_name]

    def save_index_map(self):
        import json

        index_object = self._index_map.model_dump()
        index_json = json.dumps(index_object, sort_keys=True, ensure_ascii=False)
        with open(self._index_file, "w") as fp:
            fp.write(index_json)

    def add_index(self, index_entry: RagIndexEntry):
        with self._lock:
            assert (
                len(self._index_map.indexes) < DEFAULT_MAX_INDEX_ENTRY_COUNT
            ), f"Index count should be less than {DEFAULT_MAX_INDEX_ENTRY_COUNT}."
            assert (
                index_entry.index_name not in self._index_map.indexes
            ), f"Index name '{index_entry.index_name}' already exists."
            self._index_map.indexes[index_entry.index_name] = index_entry
            self.save_index_map()
            logger.info(f"Index '{index_entry.index_name}' created successfully.")

    def update_index(self, index_entry: RagIndexEntry):
        with self._lock:
            assert (
                index_entry.index_name in self._index_map.indexes
            ), f"Index name '{index_entry.index_name}' not exists."
            self._index_map.indexes[index_entry.index_name] = index_entry
            self.save_index_map()
            logger.info(
                f"Index '{index_entry.index_name}' updated successfully {self._index_map}."
            )

    def delete_index(self, index_name: str):
        with self._lock:
            assert (
                index_name in self._index_map.indexes
            ), f"Index name '{index_name}' not exists."
            del self._index_map.indexes[index_name]
            self.save_index_map()
            logger.info(f"Index '{index_name}' removed.")

    def list_indexes(self):
        return self._index_map


index_manager = RagIndexManager.from_file(index_file=DEFAULT_INDEX_FILE)
