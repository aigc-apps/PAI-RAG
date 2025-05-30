import os
import json
import re
import shutil
import threading
from typing import Tuple, Dict, List
from pydantic import BaseModel, Field, model_validator
from pairag.core.models.state import FileServiceState
from pairag.integrations.embeddings.pai.pai_embedding_config import (
    HuggingFaceEmbeddingConfig,
)
from pairag.knowledgebase.index.pai.vector_store_config import (
    DEFAULT_LOCAL_STORAGE_PATH_OLD,
    DEFAULT_LOCAL_STORAGE_PATH,
)

# TODO: 移除knowlegebase对file的依赖
from pairag.file.nodeparsers.pai.pai_node_parser import NodeParserConfig
from pairag.knowledgebase.models import KnowledgeBase
from pairag.knowledgebase.rag_knowledgebase_helper import RagKnowledgeBaseHelper
from pairag.utils.file_utils import generate_md5
from pairag.knowledgebase.utils.knowledgebase_utils import (
    delete_dir,
    delete_knowledgebase_dir,
    delete_default_knowledgebase_dir,
)
from loguru import logger


from pairag.utils.constants import (
    DEFAULT_KNOWLEDGEBASE_NAME,
    DEFAULT_KNOWLEDGEBASE_FILE,
    DEFAULT_KNOWLEDGEBASE_PATH,
    DEFAULT_MAX_KNOWLEDGEBASE_COUNT,
    DEFAULT_KNOWLEDGEBASE_NAME_OLD,
    DEFAULT_DOC_STORE_NAME,
)
from pairag.utils.time_utils import get_current_time_str


class KnowledgeDoc(BaseModel):
    file_name: str
    doc_id: str
    file_hash: str
    last_modified_time: str = Field(default_factory=lambda x: get_current_time_str())


class KnowledgeBaseDocStore(BaseModel):
    knowledgebase: str
    doc_map: Dict[str, KnowledgeDoc] = {}


class KnowledgeBaseMap(BaseModel):
    knowledgebases: Dict[str, KnowledgeBase] = {}

    @model_validator(mode="before")
    def preprocess(cls, values: Dict) -> Dict:
        if "indexes" in values:
            values["knowledgebases"] = values["indexes"]
        return values


class KnowledgeBaseManager:
    def __init__(
        self,
        knowledgebase_file: str,
        knowledgebase_map: KnowledgeBaseMap,
    ):
        self._knowledgebase_file = knowledgebase_file
        self._knowledgebase_map = knowledgebase_map
        self._doc_store_map: Dict[str, KnowledgeBaseDocStore] = {}
        for knowlegedbase_name in self._knowledgebase_map.knowledgebases:
            self._doc_store_map[knowlegedbase_name] = self.load_doc_store(
                knowlegedbase_name
            )

        self._lock = threading.Lock()
        self._doc_lock = threading.Lock()
        self._state = FileServiceState(DEFAULT_KNOWLEDGEBASE_FILE)

    def move_old_index_persist_path(self, old_persist_path, new_persist_path):
        if os.path.exists(old_persist_path):
            if not os.path.exists(new_persist_path):
                os.makedirs(new_persist_path, exist_ok=True)
            for item in os.listdir(old_persist_path):
                source_path = os.path.join(old_persist_path, item)
                if os.path.isdir(source_path):
                    shutil.move(source_path, new_persist_path)
                    print(f"已移动目录: {source_path} 到 {new_persist_path}")
            delete_dir(old_persist_path)

    def create_default_knowledgebase(self, rag_config):
        default_knowledge_base = KnowledgeBase(
            name=DEFAULT_KNOWLEDGEBASE_NAME,
            vector_store_config=rag_config.index.vector_store,
            embedding_config=HuggingFaceEmbeddingConfig(),
            node_parser_config=NodeParserConfig(),
        )
        RagKnowledgeBaseHelper.create_new_knowledgebase_dir(DEFAULT_KNOWLEDGEBASE_NAME)
        self._knowledgebase_map.knowledgebases[
            DEFAULT_KNOWLEDGEBASE_NAME
        ] = default_knowledge_base
        new_state = self.save_knowledgebase_map()
        self._state.update_state(new_state)
        logger.info(f"创建默认知识库'{DEFAULT_KNOWLEDGEBASE_NAME}'成功。")

    def compatible_init(self, rag_config):
        if len(self._knowledgebase_map.knowledgebases) == 0:
            self.create_default_knowledgebase(rag_config)

        if DEFAULT_KNOWLEDGEBASE_NAME in self._knowledgebase_map.knowledgebases:
            if not os.path.exists(
                os.path.join(DEFAULT_KNOWLEDGEBASE_PATH, DEFAULT_KNOWLEDGEBASE_NAME)
            ):
                RagKnowledgeBaseHelper.create_new_knowledgebase_dir(
                    DEFAULT_KNOWLEDGEBASE_NAME
                )
            return

        _knowledges_cp = self._knowledgebase_map.knowledgebases.copy()
        self._knowledgebase_map.knowledgebases = {}

        if len(_knowledges_cp) > 0:
            for knowledge_name, old_knowledgebase in _knowledges_cp.items():
                new_knowledgebase_name = (
                    DEFAULT_KNOWLEDGEBASE_NAME
                    if knowledge_name == DEFAULT_KNOWLEDGEBASE_NAME_OLD
                    else knowledge_name
                )
                new_knowledgebase = KnowledgeBase(
                    name=new_knowledgebase_name,
                    vector_store_config=old_knowledgebase.vector_store_config,
                    embedding_config=old_knowledgebase.embedding_config,
                    node_parser_config=NodeParserConfig(),
                )
                RagKnowledgeBaseHelper.create_new_knowledgebase_dir(
                    new_knowledgebase_name
                )
                if old_knowledgebase.vector_store_config.type == "faiss":
                    self.move_old_index_persist_path(
                        old_knowledgebase.vector_store_config.persist_path,
                        os.path.join(
                            DEFAULT_KNOWLEDGEBASE_PATH,
                            new_knowledgebase_name,
                            ".index",
                            ".faiss",
                        ),
                    )
                new_knowledgebase.vector_store_config.persist_path = os.path.join(
                    DEFAULT_KNOWLEDGEBASE_PATH,
                    new_knowledgebase_name,
                    ".index",
                    ".faiss",
                )

                self._knowledgebase_map.knowledgebases[
                    new_knowledgebase_name
                ] = new_knowledgebase
        else:
            self.move_old_index_persist_path(
                DEFAULT_LOCAL_STORAGE_PATH_OLD,
                DEFAULT_LOCAL_STORAGE_PATH,
            )

    @classmethod
    def from_file(cls, knowledgebase_file: str):
        if os.path.exists(knowledgebase_file):
            with open(knowledgebase_file, "r") as f:
                knowledgebase_json_str = f.read()
                knowledgebase_map = KnowledgeBaseMap.model_validate_json(
                    knowledgebase_json_str
                )
        else:
            knowledgebase_map = KnowledgeBaseMap()

        return cls(
            knowledgebase_file=knowledgebase_file, knowledgebase_map=knowledgebase_map
        )

    def get_knowledgebase_map(self) -> KnowledgeBaseMap:
        return self._knowledgebase_map

    def get_knowledgebase(self, name: str = None) -> KnowledgeBase:
        if not name or name == "default_index":
            return self._knowledgebase_map.knowledgebases[DEFAULT_KNOWLEDGEBASE_NAME]

        return self._knowledgebase_map.knowledgebases[name]

    def add_knowledgebase(self, knowledgebase: KnowledgeBase):
        with self._lock:
            assert (
                len(self._knowledgebase_map.knowledgebases)
                < DEFAULT_MAX_KNOWLEDGEBASE_COUNT
            ), f"新建知识库失败: 知识库数量超过最大限制{DEFAULT_MAX_KNOWLEDGEBASE_COUNT}."
            assert (
                knowledgebase.name not in self._knowledgebase_map.knowledgebases
            ), f"新建知识库失败: 知识库'{knowledgebase.name}' 已存在。"

            self._knowledgebase_map.knowledgebases[knowledgebase.name] = knowledgebase
            RagKnowledgeBaseHelper.create_new_knowledgebase_dir(knowledgebase.name)

            new_state = self.save_knowledgebase_map()
            self._state.update_state(new_state)
            logger.info(f"知识库 '{knowledgebase.name}' 创建成功.")

    def update_knowledgebase(self, knowledgebase: KnowledgeBase):
        with self._lock:
            assert (
                knowledgebase.name in self._knowledgebase_map.knowledgebases
            ), f"更新知识库失败: 无法找到知识库'{knowledgebase.name}'."
            self._knowledgebase_map.knowledgebases[knowledgebase.name] = knowledgebase
            new_state = self.save_knowledgebase_map()
            self._state.update_state(new_state)
            logger.info(f"知识库 '{knowledgebase.name}' 更新成功.")

    def delete_knowledgebase(self, name: str):
        with self._lock:
            assert (
                name in self._knowledgebase_map.knowledgebases
            ), f"删除知识库失败: 无法找到知识库'{name}'."

            if name == DEFAULT_KNOWLEDGEBASE_NAME:
                delete_default_knowledgebase_dir()
                logger.info(f"默认知识库 '{name}' 不能被删除。本地存储已经清空。")
            else:
                del self._knowledgebase_map.knowledgebases[name]
                delete_knowledgebase_dir(name)
                logger.info(f"知识库 '{name}' 删除成功。")

            new_state = self.save_knowledgebase_map()
            self._state.update_state(new_state)

    def list_knowledgebases(self):
        return self._knowledgebase_map

    # docs api
    def get_doc_store_name(self, knowledgebase_name):
        return os.path.join(
            DEFAULT_KNOWLEDGEBASE_PATH, knowledgebase_name, DEFAULT_DOC_STORE_NAME
        )

    def load_doc_store(self, knowledgebase_name):
        if knowledgebase_name not in self._knowledgebase_map.knowledgebases:
            raise ValueError(f"知识库 '{knowledgebase_name}' 不存在。")

        doc_file_name = self.get_doc_store_name(knowledgebase_name)
        if os.path.exists(doc_file_name):
            with open(doc_file_name, "r") as f:
                doc_data = f.read()
                return KnowledgeBaseDocStore.model_validate_json(doc_data)

        return KnowledgeBaseDocStore(knowledgebase=knowledgebase_name)

    def persist_doc_store(self, doc_store: KnowledgeBaseDocStore):
        with self._doc_lock:
            doc_file_name = self.get_doc_store_name(doc_store.knowledgebase)
            doc_obj = doc_store.model_dump()
            doc_data = json.dumps(doc_obj, sort_keys=True, ensure_ascii=False)
            with open(doc_file_name, "w") as doc_f:
                doc_f.write(doc_data)
        logger.info(f"更新'{doc_store.knowledgebase}'文档库文件成功.")

    def add_doc_to_knowledgebase(
        self,
        knowledgebase_name: str,
        doc_id: str,
        file_name: str,
        file_hash: str,
        last_modified_time: str,
    ):
        if knowledgebase_name not in self._knowledgebase_map.knowledgebases:
            raise ValueError(f"知识库 '{knowledgebase_name}' 不存在。")

        doc = KnowledgeDoc(
            doc_id=doc_id,
            file_name=file_name,
            file_hash=file_hash,
            last_modified_time=last_modified_time,
        )
        if not self._doc_store_map.get(knowledgebase_name):
            self._doc_store_map[knowledgebase_name] = KnowledgeBaseDocStore(
                knowledgebase=knowledgebase_name
            )
        doc_store = self._doc_store_map.get(knowledgebase_name)
        doc_store.doc_map[doc.file_name] = doc
        self.persist_doc_store(doc_store)
        logger.info(f"文件'{doc.file_name}'成功添加到知识库'{knowledgebase_name}'。")

    def delete_doc_from_knowledgebase(self, knowledgebase_name, file_name):
        if knowledgebase_name not in self._knowledgebase_map.knowledgebases:
            raise ValueError(f"知识库 '{knowledgebase_name}' 不存在。")

        doc_store = self._doc_store_map.get(
            knowledgebase_name, KnowledgeBaseDocStore(knowledgebase=knowledgebase_name)
        )
        if file_name in doc_store.doc_map:
            del doc_store.doc_map[file_name]
            self.persist_doc_store(doc_store)
        else:
            logger.warning(f"尝试删除的文件记录'{file_name}'没有找到。")

        logger.info(f"文件'{file_name}'成功从知识库删除'{knowledgebase_name}'。")

    def get_docs_from_knowledgebase(self, knowledgebase_name: str):
        if knowledgebase_name not in self._knowledgebase_map.knowledgebases:
            raise ValueError(f"知识库 '{knowledgebase_name}' 不存在。")

        doc_store = self._doc_store_map.get(
            knowledgebase_name, KnowledgeBaseDocStore(knowledgebase=knowledgebase_name)
        )
        return list(doc_store.doc_map.values())

    """
    主要是删除的时候获取与文件相关的文档。
    """

    def get_related_docs_for_deletion(
        self, knowledgebase_name: str, file_name: str
    ) -> List[KnowledgeDoc]:
        if knowledgebase_name not in self._knowledgebase_map.knowledgebases:
            logger.warning(f"知识库 '{knowledgebase_name}' 不存在。")
            return []
        doc_store = self._doc_store_map[knowledgebase_name]
        if not os.path.isdir(file_name):
            if doc_store.doc_map.get(file_name, None):
                return [doc_store.doc_map[file_name]]

            logger.warning(f"在知识库'{knowledgebase_name}'中没找到文件'{file_name}'。")
            return []
        else:
            return [
                v
                for k, v in doc_store.doc_map.items()
                if k.startswith(f"{file_name.rstrip('/')}/")
            ]

    def check_updates(self):
        new_state = self._state.check_state()
        if new_state != 0:
            logger.info(f"检测到知识库更新 {self._state.state_key} {new_state}.")
            self.reload(new_state)
            logger.info("重新加载知识库成功.")

    def reload(self, new_state):
        with self._lock:
            if self._state.state_value != new_state:
                logger.info(
                    f"Need reload index from background. {self._state.state_key}"
                )
                if os.path.exists(DEFAULT_KNOWLEDGEBASE_FILE):
                    with open(DEFAULT_KNOWLEDGEBASE_FILE, "r") as f:
                        knowledgebase_json_str = f.read()
                        self._index_map = KnowledgeBaseMap.model_validate_json(
                            knowledgebase_json_str
                        )
                        self._state.update_state(new_state)

    # save file
    def save_knowledgebase_map(self):
        knowledgbase_map_obj = self._knowledgebase_map.model_dump()
        knowledgebase_json = json.dumps(
            knowledgbase_map_obj, sort_keys=True, ensure_ascii=False
        )
        with open(self._knowledgebase_file, "w") as fp:
            fp.write(knowledgebase_json)

        return os.path.getmtime(self._knowledgebase_file)

    def get_change_files(
        self, file_path, is_delete=False
    ) -> Tuple[str, List[KnowledgeDoc]]:
        file_path_pattern = f"^(.+/)?{DEFAULT_KNOWLEDGEBASE_PATH}/(.+?)/docs/(.+)$"
        match = re.match(file_path_pattern, file_path)
        if match:
            index_name = match.group(2)
            if index_name in self._knowledgebase_map.knowledgebases:
                if is_delete:
                    return index_name, self.get_related_docs_for_deletion(
                        index_name, file_path
                    )
                else:
                    if os.path.isfile(file_path):
                        file_path_md5, file_content_md5 = generate_md5(file_path)
                        doc = KnowledgeDoc(
                            file_name=file_path,
                            doc_id=file_path_md5,
                            file_hash=file_content_md5,
                        )
                        if index_name in self._doc_store_map:
                            _doc_map = self._doc_store_map[index_name].doc_map
                            if (
                                file_path in _doc_map
                                and _doc_map[file_path].file_hash == file_content_md5
                            ):
                                return index_name, []
                        return index_name, [doc]
                    else:
                        return index_name, []
        return None, []


knowledgebase_manager = KnowledgeBaseManager.from_file(
    knowledgebase_file=DEFAULT_KNOWLEDGEBASE_FILE
)
