from abc import ABC, abstractmethod
from typing import List
from loguru import logger

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.schema import QueryBundle

from pairag.knowledgebase.index.pai.pai_vector_index import PaiVectorStoreIndex
from pairag.integrations.data_analysis.text2sql.db_info_index import (
    SchemaIndex,
    HistoryIndex,
    ValueIndex,
)


# 数据库信息（向量）检索接口
class DBInfoRetriever(ABC):
    """
    Abstract base class for database information vector-based retrieval.
    """

    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        self._db_name = db_name
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k

    @abstractmethod
    def get_index(self, nodes: List[TextNode]):
        pass

    @abstractmethod
    def retrieve_nodes(self, query: QueryBundle | str | List):
        pass

    @abstractmethod
    async def aretrieve_nodes(self, query: QueryBundle | str | List):
        pass


class SchemaRetriever(DBInfoRetriever):
    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        super().__init__(db_name, embed_model, similarity_top_k)
        self._schema_index = SchemaIndex(
            db_name=db_name, embed_model=embed_model, similarity_top_k=similarity_top_k
        )
        self._schema_retriever = self._schema_index.as_retriever()

    def get_index(self, nodes: List[TextNode]):
        # 先删除（更新顺序后）再插入
        del_nodes(self._schema_index._description_index, nodes)
        self._schema_index.insert_nodes(nodes)
        # status = update_docstore(self._schema_index._description_index, nodes)
        # if status  == "no update":
        #     return self._schema_index.insert_nodes(nodes)

    def retrieve_nodes(self, query):
        retrieved_nodes = self._schema_retriever.retrieve(query)

        return retrieved_nodes

    async def aretrieve_nodes(self, query):
        retrieved_nodes = await self._schema_retriever.aretrieve(query)

        return retrieved_nodes


class HistoryRetriever(DBInfoRetriever):
    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        super().__init__(db_name, embed_model, similarity_top_k)
        self._history_index = HistoryIndex(
            db_name=db_name, embed_model=embed_model, similarity_top_k=similarity_top_k
        )
        self._history_retriever = self._history_index.as_retriever()

    def get_index(self, nodes: List[TextNode]):
        del_nodes(self._history_index._history_index, nodes)
        self._history_index.insert_nodes(nodes)
        # status = update_docstore(self._history_index._history_index, nodes)
        # if status  == "no update":
        #     return self._history_index.insert_nodes(nodes)

    def retrieve_nodes(self, query):
        retrieved_nodes = self._history_retriever.retrieve(query)

        return retrieved_nodes

    async def aretrieve_nodes(self, query):
        retrieved_nodes = await self._history_retriever.aretrieve(query)

        return retrieved_nodes


class ValueRetriever(DBInfoRetriever):
    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        super().__init__(db_name, embed_model, similarity_top_k)
        self._value_index = ValueIndex(
            db_name=db_name, embed_model=embed_model, similarity_top_k=similarity_top_k
        )
        self._value_retriever = self._value_index.as_retriever()

    def get_index(self, nodes: List[TextNode]):
        del_nodes(self._value_index._value_index, nodes)
        self._value_index.insert_nodes(nodes)
        # status = update_docstore(self._value_index._value_index, nodes)
        # if status  == "no update":
        #     return self._value_index.insert_nodes(nodes)

    def retrieve_nodes(self, query):
        retrieved_nodes_list = []
        if len(query) != 0:
            for keyword in query:
                retrieved_nodes = self._value_retriever.retrieve(keyword)
                retrieved_nodes_list.extend(retrieved_nodes)

        return retrieved_nodes_list

    async def aretrieve_nodes(self, query):
        retrieved_nodes_list = []
        if len(query) != 0:
            for keyword in query:
                retrieved_nodes = await self._value_retriever.aretrieve(keyword)
                retrieved_nodes_list.extend(retrieved_nodes)

        return retrieved_nodes_list


def delete_nodes_index(vector_index: PaiVectorStoreIndex, nodes: List[TextNode]):
    """
    Depreciated
    删除docstore中的doc,
    删除index_struct.nodes_dict中的node_id
    删除faiss_index中的id
    """
    existing_node_ids = [
        node.id_ for node in vector_index.storage_context.docstore.docs.values()
    ]
    new_node_ids = [node.id_ for node in nodes]
    remaining_node_ids = list(set(existing_node_ids) - set(new_node_ids))
    logger.debug(f"existing_node_ids: {existing_node_ids}")
    logger.debug(f"new_node_ids: {new_node_ids}")
    logger.debug(f"remaining_node_ids: {remaining_node_ids}")

    if remaining_node_ids:
        # 删除docstore中的doc并更新引用
        for node_id in remaining_node_ids:
            vector_index.storage_context.docstore.delete_document(node_id)
            # print("length:", len(index.storage_context.docstore.docs.values()))
        # 保存更改
        vector_index.storage_context.persist(vector_index._persist_path)
        # 更新索引结构中的节点引用
        new_nodes_dict = {}
        remaining_keys = []
        for k, v in vector_index._vector_index.index_struct.nodes_dict.items():
            if v not in remaining_node_ids:
                new_nodes_dict[k] = v
            else:
                remaining_keys.append(k)
        # print("new_nodes_dict:", new_nodes_dict)
        # print("remaining_keys:", remaining_keys)
        vector_index._vector_index.index_struct.nodes_dict = new_nodes_dict
        # 删除faiss中的id
        vector_index._vector_store.remove_ids(remaining_keys)


# def update_docstore(
#     vector_index: PaiVectorStoreIndex, nodes: List[TextNode], is_history: bool = False
# ):
#     existing_node_ids = [
#         node.id_ for node in vector_index.storage_context.docstore.docs.values()
#     ]
#     new_node_ids = [node.id_ for node in nodes]
#     existing_difference_ids = list(set(existing_node_ids) - set(new_node_ids))
#     new_difference_ids = list(set(new_node_ids) - set(existing_node_ids))
#     # logger.debug(f"existing_node_ids: {existing_node_ids}")
#     # logger.debug(f"new_node_ids: {new_node_ids}")
#     # logger.debug(f"remaining_node_ids: {remaining_node_ids}")

#     if existing_difference_ids:
#         assert len(existing_difference_ids) == len(new_difference_ids)
#         # 获取existing_difference_ids对应的node,并构造查询dict
#         lookup_dict = {}
#         for node_id in existing_difference_ids:
#             existing_node = vector_index.storage_context.docstore.get_document(node_id)
#             if is_history:
#                 lookup_dict[existing_node.metadata["query"]] = node_id
#             else:
#                 lookup_dict[
#                     (
#                         existing_node.metadata["table_name"],
#                         existing_node.metadata["column_name"],
#                     )
#                 ] = node_id
#             # lookup_dict[(existing_node.metadata["table_name"], existing_node.metadata["column_name"])] = node_id
#         for node in nodes:
#             if node.id_ in new_difference_ids:
#                 if is_history:
#                     node.id_ = lookup_dict[existing_node.metadata["query"]]
#                 else:
#                     node.id_ = lookup_dict[
#                         (node.metadata["table_name"], node.metadata["column_name"])
#                     ]

#                 vector_index.storage_context.docstore.add_documents(
#                     [node], allow_update=True
#                 )

#         # 保存更改
#         vector_index.storage_context.persist(vector_index._persist_path)

#         return "updated"

#     else:
#         return "no update"


def del_nodes(vector_index: PaiVectorStoreIndex, nodes: List[TextNode]):
    # 对比查找待删除nodes
    existing_node_ids = [
        node.id_ for node in vector_index.storage_context.docstore.docs.values()
    ]
    new_node_ids = [node.id_ for node in nodes]
    to_del_node_ids = list(set(existing_node_ids) - set(new_node_ids))

    if to_del_node_ids:
        # 删除docstore中的doc并更新引用
        for node_id in to_del_node_ids:
            vector_index.storage_context.docstore.delete_document(node_id)

        # 按键的数值顺序排序索引结构
        sorted_items = sorted(
            vector_index._vector_index.index_struct.nodes_dict.items(),
            key=lambda item: int(item[0]),
        )
        # 更新索引结构的节点引用
        remaining_nodes_dict = {}
        remaining_key = 0
        to_del_keys = []
        for k, v in sorted_items:
            if v not in to_del_node_ids:
                remaining_nodes_dict[str(remaining_key)] = v
                remaining_key += 1
            else:
                to_del_keys.append(k)

        logger.debug("remaining_nodes_dict:", remaining_nodes_dict)
        vector_index._vector_index.index_struct.nodes_dict = remaining_nodes_dict
        vector_index.storage_context.index_store.add_index_struct(
            vector_index._vector_index.index_struct
        )
        # 保存更改
        vector_index.storage_context.persist(vector_index._persist_path)
        # 删除faiss中的id
        vector_index._vector_store.remove_ids(to_del_keys)
