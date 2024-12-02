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
from pai_rag.integrations.data_analysis.nl2sql.db_utils.nl2sql_utils import (
    count_total_columns,
    get_target_info,
    extract_subset_from_description,
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

        db_description_dict = get_target_info(
            self._db_description_path, db_description_dict, "description"
        )
        total_columns = count_total_columns(db_description_dict)

        # 检索description返回List[NodeWithScore]
        retrieved_description_nodes = self._retrieve_context_nodes(
            self._db_description_index, nl_query, top_k
        )
        logger.info(
            f"Description nodes retrieved from index, number of nodes: {len(retrieved_description_nodes), retrieved_description_nodes}"
        )
        # 检索entity返回List[NodeWithScore]
        retrieved_value_nodes = self._retrieve_entity_nodes(
            self._db_value_index, keywords, top_k
        )
        logger.info(
            f"Value nodes retrieved from index, number of nodes: {len(retrieved_value_nodes), retrieved_value_nodes}."
        )
        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_description_dict = self._filter_description(
            retrieved_description_nodes, retrieved_value_nodes, db_description_dict
        )
        retrieved_total_columns = count_total_columns(retrieved_description_dict)
        logger.info(
            f"Description dict filtered, number from {total_columns} to {retrieved_total_columns}."
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

        db_description_dict = get_target_info(
            self._db_description_path, db_description_dict, "description"
        )
        total_columns = count_total_columns(db_description_dict)

        # 检索返回List[NodeWithScore]
        retrieved_description_nodes = await self._aretrieve_context_nodes(
            self._db_description_index, nl_query, top_k
        )
        logger.info(
            f"Description nodes retrieved from index, number of nodes: {len(retrieved_description_nodes), retrieved_description_nodes}"
        )
        # 检索entity返回List[NodeWithScore]
        retrieved_value_nodes = await self._aretrieve_entity_nodes(
            self._db_value_index, keywords, top_k
        )
        logger.info(
            f"Value nodes retrieved from index, number of nodes: {len(retrieved_value_nodes), retrieved_value_nodes}"
        )

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_description_dict = self._filter_description(
            retrieved_description_nodes, retrieved_value_nodes, db_description_dict
        )
        retrieved_total_columns = count_total_columns(retrieved_description_dict)
        logger.info(
            f"Description dict filtered, number from {total_columns} to {retrieved_total_columns}."
        )

        return retrieved_description_dict

    def get_retrieved_history(
        self,
        nl_query: QueryBundle,
        top_k: int = 10,
        db_history_list: Optional[List] = None,
    ) -> List[Dict]:
        """get retrieved query history"""

        db_history_list = get_target_info(
            self._db_description_path, db_history_list, "history"
        )
        history_nums = len(db_history_list)

        # 检索返回List[NodeWithScore]
        retrieved_history_nodes = self._retrieve_context_nodes(
            self._db_history_index, nl_query, top_k
        )
        logger.info(
            f"History nodes retrieved from index, number of nodes: {len(retrieved_history_nodes)}"
        )

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_history_list = self._filter_history(
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

        db_history_list = get_target_info(
            self._db_history_path, db_history_list, "history"
        )
        history_nums = len(db_history_list)

        # 检索返回List[NodeWithScore]
        retrieved_history_nodes = await self._aretrieve_context_nodes(
            self._db_history_index, nl_query, top_k
        )
        logger.info(
            f"History nodes retrieved from index, number of nodes: {len(retrieved_history_nodes)}"
        )

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_history_list = self._filter_history(
            retrieved_history_nodes, db_history_list
        )
        logger.info(
            f"History list filtered, number from {history_nums} to {len(retrieved_history_list)}"
        )

        return retrieved_history_list

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

    def _filter_description(
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
            logger.info(
                f"retrieved_nodes_dict from value_nodes: {len(retrieved_nodes_dict)},\n {retrieved_nodes_dict}"
            )
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

        # 过滤db_description_dict
        filterd_db_description_dict = extract_subset_from_description(
            retrieved_nodes_dict, db_description_dict
        )

        return filterd_db_description_dict

    def _filter_history(
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
