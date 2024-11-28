import os
import json
from typing import Optional, Dict, List
from loguru import logger

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

from pai_rag.integrations.data_analysis.nl2sql.db_utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
    DEFAULT_DB_HISTORY_PATH,
    DESCRIPTION_STORAGE_PATH,
    HISTORY_STORAGE_PATH,
    VALUE_STORAGE_PATH,
)


class DBPreRetriever:
    """
    基于自然语言问题、关键词等进行预检索
    """

    def __init__(
        self,
        keywords: Optional[List] = None,
        embed_model: Optional[BaseEmbedding] = None,
        db_description_path: Optional[str] = None,
        db_history_path: Optional[str] = None,
        description_storage_path: Optional[str] = None,
        history_storage_path: Optional[str] = None,
        value_storage_path: Optional[str] = None,
    ) -> None:
        self._keywords = keywords
        self._embed_model = embed_model or Settings.embed_model
        self._db_description_path = db_description_path or DEFAULT_DB_DESCRIPTION_PATH
        self._db_history_path = db_history_path or DEFAULT_DB_HISTORY_PATH
        self._description_storage_path = (
            description_storage_path or DESCRIPTION_STORAGE_PATH
        )
        self._history_storage_path = history_storage_path or HISTORY_STORAGE_PATH
        self._value_storage_path = value_storage_path or VALUE_STORAGE_PATH
        self._db_description_index = self._load_index(self._description_storage_path)
        self._db_history_index = self._load_index(self._history_storage_path)
        self._db_value_index = self._load_index(self._value_storage_path)

    def _load_index(self, file_path: str) -> VectorStoreIndex:
        """从本地加载索引"""
        try:
            loaded_storage_context = StorageContext.from_defaults(persist_dir=file_path)
            loaded_index = load_index_from_storage(
                loaded_storage_context, embed_model=self._embed_model
            )
            logger.info(f"Index loaded from {file_path}")
        except Exception as e:
            loaded_index = None
            logger.info(f"Index loaded from {file_path} failed: {e}")

        return loaded_index

    def get_retrieved_description(
        self,
        nl_query: QueryBundle,
        keywords: List,
        top_k: int = 10,
        db_description_dict: Optional[Dict] = None,
    ) -> Dict:
        """get retrieved schema description"""

        db_description_dict = self._get_target_info(db_description_dict, "description")
        column_nums = len(db_description_dict["column_info"])

        # 检索description返回List[NodeWithScore]
        retrieved_description_nodes = self._retrieve_context_nodes(
            self._db_description_index, nl_query, top_k
        )
        logger.info(
            f"Description nodes retrieved from index, number of nodes: {len(retrieved_description_nodes)}"
        )
        # 检索entity返回List[NodeWithScore]
        retrieved_value_nodes = self._retrieve_entity_nodes(
            self._db_value_index, keywords, top_k
        )
        logger.info(
            f"Value nodes retrieved from index, number of nodes: {len(retrieved_value_nodes)}"
        )

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_description_dict = self._description_filter(
            retrieved_description_nodes, retrieved_value_nodes, db_description_dict
        )
        logger.info(
            f"""Description dict filtered, number from {column_nums} to {len(retrieved_description_dict["column_info"])}"""
        )

        return retrieved_description_dict

    async def aget_retrieved_description(
        self,
        nl_query: QueryBundle,
        keywords: List,
        top_k: int = 10,
        db_description_dict: Optional[Dict] = None,
    ) -> Dict:
        """get retrieved schema description"""

        db_description_dict = self._get_target_info(db_description_dict, "description")
        column_nums = len(db_description_dict["column_info"])

        # 检索返回List[NodeWithScore]
        retrieved_description_nodes = await self._aretrieve_context_nodes(
            self._db_description_index, nl_query, top_k
        )
        logger.info(
            f"Description nodes retrieved from index, number of nodes: {len(retrieved_description_nodes)}"
        )
        # 检索entity返回List[NodeWithScore]
        retrieved_value_nodes = await self._aretrieve_entity_nodes(
            self._db_value_index, keywords, top_k
        )
        logger.info(
            f"Value nodes retrieved from index, number of nodes: {len(retrieved_value_nodes)}"
        )

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_description_dict = self._description_filter(
            retrieved_description_nodes, retrieved_value_nodes, db_description_dict
        )
        logger.info(
            f"""Description dict filtered, number from {column_nums} to {len(retrieved_description_dict["column_info"])}"""
        )

        return retrieved_description_dict

    def get_retrieved_history(
        self,
        nl_query: QueryBundle,
        top_k: int = 10,
        db_history_list: Optional[List] = None,
    ) -> List[Dict]:
        """get retrieved query history"""

        db_history_list = self._get_target_info(db_history_list, "history")
        history_nums = len(db_history_list)

        # 检索返回List[NodeWithScore]
        retrieved_history_nodes = self._retrieve_context_nodes(
            self._db_history_index, nl_query, top_k
        )
        logger.info(
            f"History nodes retrieved from index, number of nodes: {len(retrieved_history_nodes)}"
        )

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_history_list = self._history_filter(
            retrieved_history_nodes, db_history_list
        )
        logger.info(
            f"History list filtered, number from {history_nums} to {len(retrieved_history_list)}"
        )

        return retrieved_history_list

    async def aget_retrieved_history(
        self,
        nl_query: QueryBundle,
        top_k: int = 10,
        db_history_list: Optional[List] = None,
    ) -> List[Dict]:
        """get retrieved query history"""

        db_history_list = self._get_target_info(db_history_list, "history")
        history_nums = len(db_history_list)

        # 检索返回List[NodeWithScore]
        retrieved_history_nodes = await self._aretrieve_context_nodes(
            self._db_history_index, nl_query, top_k
        )
        logger.info(
            f"History nodes retrieved from index, number of nodes: {len(retrieved_history_nodes)}"
        )

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_history_list = self._history_filter(
            retrieved_history_nodes, db_history_list
        )
        logger.info(
            f"History list filtered, number from {history_nums} to {len(retrieved_history_list)}"
        )

        return retrieved_history_list

    def _get_target_info(
        self, target: Optional[Dict | List] = None, flag: str = "description"
    ) -> Dict | List:
        # 正常情况下接受传入的description dict 或 history list，否则从本地加载
        if target is None:
            if flag == "description":
                if os.path.exists(self._db_description_path):
                    file_path = self._db_description_path
                else:
                    raise ValueError(
                        f"db_description_file_path: {self._db_description_path} does not exist"
                    )
            if flag == "history":
                if os.path.exists(self._db_history_path):
                    file_path = self._db_history_path
                else:
                    raise ValueError(
                        f"db_history_file_path: {self._db_history_path} does not exist"
                    )
            try:
                with open(file_path, "r") as f:
                    target = json.load(f)
            except Exception as e:
                raise ValueError(f"Load target object from {file_path} failed: {e}")
                # logger.info(f"Load target object from {file_path} failed: {e}")

        return target

    def _retrieve_context_nodes(
        self,
        index: VectorStoreIndex,
        nl_query: QueryBundle,
        top_k: int,
    ) -> List[NodeWithScore]:
        """used for retrieve top k description or history nodes"""
        if index:
            retriever = index.as_retriever(similarity_top_k=top_k)
            retrieved_nodes = retriever.retrieve(nl_query)
        else:
            retrieved_nodes = []

        return retrieved_nodes

    async def _aretrieve_context_nodes(
        self,
        index: VectorStoreIndex,
        nl_query: QueryBundle,
        top_k: int,
    ) -> List[NodeWithScore]:
        """used for retrieve top k description or history nodes"""
        if index:
            retriever = index.as_retriever(similarity_top_k=top_k)
            retrieved_nodes = await retriever.aretrieve(nl_query)
        else:
            retrieved_nodes = []

        return retrieved_nodes

    # def _description_filter(
    #     self,
    #     retrieved_description_nodes: List[NodeWithScore],
    #     db_description_dict: Dict,
    # ) -> Dict:
    #     """根据retrieve结果缩小db_description中column_info"""

    #     if len(retrieved_description_nodes) == 0:
    #         logger.info(
    #             "Empty retrieved_description_nodes, use original description instead."
    #         )
    #         return db_description_dict

    #     else:
    #         retrieved_nodes_list = []
    #         for node in retrieved_description_nodes:
    #             retrieved_nodes_list.append(
    #                 {
    #                     "table": node.metadata["table_name"],
    #                     "column": node.metadata["column_name"],
    #                 }
    #             )

    #         retrieved_db_description_list = []
    #         for item in db_description_dict["column_info"]:
    #             if {
    #                 "table": item["table"],
    #                 "column": item["column"],
    #             } in retrieved_nodes_list:
    #                 retrieved_db_description_list.append(item)
    #             if (item["primary_key"] is True or item["foreign_key"] is True) and (
    #                 item not in retrieved_db_description_list
    #             ):  # 保留主键和外键
    #                 retrieved_db_description_list.append(item)
    #         # update with selected_db_description_list
    #         db_description_dict["column_info"] = retrieved_db_description_list

    #         return db_description_dict

    def _description_filter(
        self,
        retrieved_description_nodes: List[NodeWithScore],
        retrieved_value_nodes: List[NodeWithScore],
        db_description_dict: Dict,
    ) -> Dict:
        """根据retrieved description_nodes和value_nodes进行过滤，缩小db_description_dict"""

        # 从retrieved_value_nodes和retrieved_description_nodes中获取retrieved_nodes_dict
        retrieved_nodes_dict = {}
        if len(retrieved_value_nodes) > 0:
            for node in retrieved_value_nodes:
                key = (node.metadata["table_name"], node.metadata["column_name"])
                value = node.text
                if key not in retrieved_nodes_dict:
                    retrieved_nodes_dict[key] = [str(value)]
                else:
                    retrieved_nodes_dict[key].append(str(value))
        else:
            logger.info("Empty retrieved_value_nodes")
        if len(retrieved_description_nodes) > 0:
            for node in retrieved_description_nodes:
                key = (node.metadata["table_name"], node.metadata["column_name"])
                if key not in retrieved_nodes_dict:
                    retrieved_nodes_dict[key] = []
            logger.info(
                f"retrieved_nodes_dict: {len(retrieved_nodes_dict)},\n {retrieved_nodes_dict}"
            )
        else:
            logger.info("Empty retrieved_description_nodes")

        # 从db_description_dict中获取满足过滤key的column_info
        if len(retrieved_nodes_dict) > 0:
            retrieved_db_description_list = []
            for item in db_description_dict["column_info"]:
                key = (item["table"], item["column"])
                if key in retrieved_nodes_dict:
                    if len(retrieved_nodes_dict[key]) > 0:
                        item["value_sample"].extend(retrieved_nodes_dict[key])
                        item["value_sample"] = list(set(item["value_sample"]))
                    retrieved_db_description_list.append(item)
                if (item["primary_key"] is True or item["foreign_key"] is True) and (
                    item not in retrieved_db_description_list
                ):  # 保留主键和外键
                    retrieved_db_description_list.append(item)

            # update with retrieved_db_description_list
            db_description_dict["column_info"] = retrieved_db_description_list

        return db_description_dict

    def _history_filter(
        self, retrieved_history_nodes: List[NodeWithScore], db_history_list: List
    ) -> List:
        """根据retrieve结果缩小db_history"""

        if len(retrieved_history_nodes) == 0:
            logger.info("Empty retrieved_history_nodes, use original history instead.")
            return db_history_list

        else:
            retrieved_nodes_list = []
            for node in retrieved_history_nodes:
                retrieved_nodes_list.append({"query": node.metadata["query"]})

            retrieved_db_history_list = []
            for item in db_history_list:
                # 检查 item['query'] 是否在 retrieved_nodes_list 中
                if any(
                    item["query"] == filter_item["query"]
                    for filter_item in retrieved_nodes_list
                ):
                    retrieved_db_history_list.append(item)

            return retrieved_db_history_list

    def _value_filter(
        self,
    ):
        pass

    def _retrieve_entity_nodes(
        self, index: VectorStoreIndex, keywords: List[str], top_k: int
    ) -> List:
        """used for retrieve db value nodes"""
        retrieved_value_nodes = []

        if index and len(keywords) != 0:
            retriever = index.as_retriever(similarity_top_k=top_k)
            for keyword in keywords:
                retrieved_nodes = retriever.retrieve(keyword)
                retrieved_value_nodes.extend(retrieved_nodes)

        return retrieved_value_nodes

    async def _aretrieve_entity_nodes(
        self, index: VectorStoreIndex, keywords: List[str], top_k: int
    ) -> List:
        """used for retrieve db value nodes"""
        retrieved_value_nodes = []

        if index and len(keywords) != 0:
            retriever = index.as_retriever(similarity_top_k=top_k)
            for keyword in keywords:
                retrieved_nodes = await retriever.aretrieve(keyword)
                retrieved_value_nodes.extend(retrieved_nodes)

        return retrieved_value_nodes
