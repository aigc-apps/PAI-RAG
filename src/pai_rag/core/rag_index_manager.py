import os
from threading import Lock
import threading
from typing import Annotated, Union, Dict, List
from pydantic import BaseModel, Field
from pai_rag.core.models.state import FileServiceState
from pai_rag.core.rag_config import RagConfig
from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
)
from pai_rag.integrations.index.pai.vector_store_config import BaseVectorStoreConfig
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.embeddings.pai.embedding_utils import create_embedding
from pai_rag.core.rag_knowledgebase_manager import RagKnowledgeBaseManager
from pai_rag.utils.index_utils import delete_index_dir
from loguru import logger


DEFAULT_INDEX_FILE = "localdata/default__rag__index.json"
DEFAULT_INDEX_NAME = "default_index"
DEFAULT_MAX_INDEX_ENTRY_COUNT = os.environ.get("DEFAULT_MAX_INDEX_ENTRY_COUNT", 20)
IGNORE_FILE_LIST = [".DS_Store"]

# 共享的批处理文件列表和锁
batch_files: Dict[str, List[str]] = {}
batch_lock = Lock()

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
    knowledgebase_manager: RagKnowledgeBaseManager = RagKnowledgeBaseManager()


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
        self._state = FileServiceState(DEFAULT_INDEX_FILE)

    def add_default_index(self, rag_config: RagConfig):
        if DEFAULT_INDEX_NAME not in self._index_map.indexes:
            self._index_map.indexes[DEFAULT_INDEX_NAME] = RagIndexEntry(
                index_name=DEFAULT_INDEX_NAME,
                vector_store_config=rag_config.index.vector_store,
                embedding_config=rag_config.embedding,
                knowledgebase_manager=RagKnowledgeBaseManager(),
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

        if index_name not in self._index_map.indexes:
            new_state = self._state.check_state()
            self.reload_indexes(
                new_state=new_state
            )  # try to reload index if index not exists.
            if index_name not in self._index_map.indexes:
                raise ValueError(f"Index name '{index_name}' not exists.")
        return self._index_map.indexes[index_name]

    def save_index_map(self):
        import json

        index_object = self._index_map.model_dump()
        index_json = json.dumps(index_object, sort_keys=True, ensure_ascii=False)
        with open(self._index_file, "w") as fp:
            fp.write(index_json)

        return os.path.getmtime(self._index_file)

    def add_index(self, index_entry: RagIndexEntry):
        with self._lock:
            assert (
                len(self._index_map.indexes) < DEFAULT_MAX_INDEX_ENTRY_COUNT
            ), f"Index count should be less than {DEFAULT_MAX_INDEX_ENTRY_COUNT}."
            assert (
                index_entry.index_name not in self._index_map.indexes
            ), f"Index name '{index_entry.index_name}' already exists."
            self._index_map.indexes[index_entry.index_name] = index_entry
            new_state = self.save_index_map()
            self._state.update_state(new_state)
            logger.info(f"Index '{index_entry.index_name}' created successfully.")

    def update_index(self, index_entry: RagIndexEntry):
        with self._lock:
            assert (
                index_entry.index_name in self._index_map.indexes
            ), f"Index name '{index_entry.index_name}' not exists."
            self._index_map.indexes[index_entry.index_name] = index_entry
            new_state = self.save_index_map()
            self._state.update_state(new_state)
            logger.info(
                f"Index '{index_entry.index_name}' updated successfully {self._index_map}."
            )

    def delete_index(self, index_name: str):
        with self._lock:
            assert (
                index_name in self._index_map.indexes
            ), f"Index name '{index_name}' not exists."
            del self._index_map.indexes[index_name]
            delete_index_dir(index_name)
            new_state = self.save_index_map()
            self._state.update_state(new_state)
            logger.info(f"Index '{index_name}' removed.")

    def list_indexes(self):
        return self._index_map

    def check_updates(self):
        new_state = self._state.check_state()
        if new_state != 0:
            logger.info(
                f"Detected changes for index file {self._state.state_key} {new_state}."
            )
            self.reload_indexes(new_state)
            logger.info("Indexes reloaded successfully.")

    def reload_indexes(self, new_state):
        with self._lock:
            if self._state.state_value != new_state:
                logger.info(
                    f"Need reload index from background. {self._state.state_key}"
                )
                if os.path.exists(DEFAULT_INDEX_FILE):
                    with open(DEFAULT_INDEX_FILE, "r") as f:
                        index_json_str = f.read()
                        self._index_map = RagIndexMap.model_validate_json(
                            index_json_str
                        )
                        self._state.update_state(new_state)

    def add_file_to_index(self, index_name: str, file_path: str):
        for ignore_file in IGNORE_FILE_LIST:
            if file_path.endswith(ignore_file):
                logger.info(f"File {file_path} is not supported and ignored.")
                return
        with batch_lock:
            if index_name in batch_files:
                batch_files[index_name].append(file_path)
            else:
                batch_files[index_name] = [file_path]
        logger.info(
            f"File {file_path} added to batch_processor for index {index_name}."
        )

    def delete_file_from_index(self, index_name: str, file_path: str):
        for ignore_file in IGNORE_FILE_LIST:
            if file_path.endswith(ignore_file):
                logger.info(f"File {file_path} is not supported and ignored.")
                return True
        current_index = self.get_index_by_name(index_name)
        current_vector_store_index = PaiVectorStoreIndex(
            current_index.vector_store_config,
            embed_model=create_embedding(current_index.embedding_config),
        )
        ref_doc_id = (
            current_index.knowledgebase_manager.get_docid_from_index_via_file_name(
                file_path
            )
        )
        logger.info(
            f"get_docid_from_index_via_file_name: ref_doc_id {ref_doc_id} file path: {file_path}"
        )
        try:
            res = current_vector_store_index.delete_ref_doc(ref_doc_id)
            current_index.knowledgebase_manager.delete_local_files_from_index(file_path)
            logger.info(
                f"File {file_path} removed from batch_processor for index {index_name}. res: {res}."
            )
            return True
        except NotImplementedError as e:
            logger.error(f"Deletion not implemented: {e}")
        except Exception as e:
            logger.error(f"delete_file_from_index: delete_ref_doc error {e}")
        return False

    def delete_dir_from_index(self, index_name: str, file_path: str):
        current_index = self.get_index_by_name(index_name)
        current_index.knowledgebase_manager.delete_local_dir_from_index(file_path)


index_manager = RagIndexManager.from_file(index_file=DEFAULT_INDEX_FILE)
