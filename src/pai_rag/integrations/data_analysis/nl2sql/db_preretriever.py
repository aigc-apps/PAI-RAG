import os
import pickle
from typing import Optional, Dict, List, Tuple, Any
from loguru import logger
from datasketch import MinHash, MinHashLSH

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex

from pai_rag.integrations.data_analysis.nl2sql.db_utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
    DEFAULT_DB_HISTORY_PATH,
    DESCRIPTION_STORAGE_PATH,
    HISTORY_STORAGE_PATH,
    VALUE_STORAGE_PATH,
    VALUE_LSH_PATH,
)
from pai_rag.integrations.data_analysis.nl2sql.db_utils.nl2sql_utils import (
    count_total_columns,
    get_target_info,
    extract_subset_from_description,
    create_minhash,
    jaccard_similarity,
)
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex


class DBPreRetriever:
    """
    基于自然语言问题、关键词等进行预检索
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        description_index: PaiVectorStoreIndex,
        history_index: PaiVectorStoreIndex,
        value_index: PaiVectorStoreIndex,
        db_description_path: Optional[str] = None,
        db_history_path: Optional[str] = None,
        description_storage_path: Optional[str] = None,
        history_storage_path: Optional[str] = None,
        value_storage_path: Optional[str] = None,
        value_lsh_path: Optional[str] = None,
    ) -> None:
        self._embed_model = embed_model
        self._description_index = description_index
        self._history_index = history_index
        self._value_index = value_index

        self._db_description_path = db_description_path or DEFAULT_DB_DESCRIPTION_PATH
        self._db_history_path = db_history_path or DEFAULT_DB_HISTORY_PATH
        self._description_storage_path = (
            description_storage_path or DESCRIPTION_STORAGE_PATH
        )
        self._history_storage_path = history_storage_path or HISTORY_STORAGE_PATH
        self._value_storage_path = value_storage_path or VALUE_STORAGE_PATH
        self._value_lsh_path = value_lsh_path or VALUE_LSH_PATH
        # self._vector_store, self._storage_context = get_vector_store(embed_model_type, vector_store_type)
        # self._db_description_index = self._load_index(self._description_storage_path)
        # self._db_history_index = self._load_index(self._history_storage_path)
        # self._db_value_index = self._load_index(self._value_storage_path)
        # self._db_value_lsh, self._db_value_minhashes = self._load_lsh(
        #     self._value_lsh_path
        # )

    # def _load_index(self, file_path: str) -> VectorStoreIndex:
    #     """从本地加载索引"""
    #     try:
    #         loaded_storage_context = StorageContext.from_defaults(vector_store=self._vector_store, persist_dir=file_path)
    #         loaded_index = load_index_from_storage(
    #             loaded_storage_context, embed_model=self._embed_model
    #         )
    #         logger.info(f"Index loaded from {file_path}")
    #     except Exception as e:
    #         loaded_index = None
    #         logger.warning(f"Index not loaded from {file_path}: {e}")

    #     return loaded_index

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

        # db_description_index = self._load_index(self._description_storage_path)
        # 检索description返回List[NodeWithScore]
        retrieved_description_nodes = self._retrieve_context_nodes(
            self._description_index, nl_query, top_k
        )
        logger.info(
            f"Description nodes retrieved from index, number of nodes: {len(retrieved_description_nodes), retrieved_description_nodes}"
        )
        # db_value_index = self._load_index(self._value_storage_path)
        # 检索entity返回List[NodeWithScore]
        retrieved_value_nodes = self._retrieve_entity_nodes(
            self._value_index, keywords, top_k
        )
        logger.info(
            f"Value nodes retrieved from index, number of nodes: {len(retrieved_value_nodes), retrieved_value_nodes}."
        )
        # LSH检索entity返回List
        # similar_entities_via_LSH = self._get_similar_entities_via_LSH(keywords)
        # logger.info(
        #     f"Value item retrieved from LSH, number of items: {len(similar_entities_via_LSH), similar_entities_via_LSH}."
        # )
        similar_entities_via_LSH = []

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_description_dict = self._filter_description(
            retrieved_description_nodes,
            retrieved_value_nodes,
            similar_entities_via_LSH,
            db_description_dict,
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

        # db_description_index = self._load_index(self._description_storage_path)
        # 检索返回List[NodeWithScore]
        retrieved_description_nodes = await self._aretrieve_context_nodes(
            self._description_index, nl_query, top_k
        )
        logger.info(
            f"Description nodes retrieved from index, number of nodes: {len(retrieved_description_nodes), retrieved_description_nodes}"
        )
        # db_value_index = self._load_index(self._value_storage_path)
        # 检索entity返回List[NodeWithScore]
        retrieved_value_nodes = await self._aretrieve_entity_nodes(
            self._value_index, keywords, top_k
        )
        logger.info(
            f"Value nodes retrieved from index, number of nodes: {len(retrieved_value_nodes), retrieved_value_nodes}"
        )
        # LSH检索entity返回List
        # similar_entities_via_LSH = self._get_similar_entities_via_LSH(keywords)
        # logger.info(
        #     f"Value item retrieved from LSH, number of items: {len(similar_entities_via_LSH), similar_entities_via_LSH}."
        # )
        similar_entities_via_LSH = []

        # 如有检索结果，进一步缩小db_description_dict，否则返回原始dict
        retrieved_description_dict = self._filter_description(
            retrieved_description_nodes,
            retrieved_value_nodes,
            similar_entities_via_LSH,
            db_description_dict,
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
            self._db_history_path, db_history_list, "history"
        )
        history_nums = len(db_history_list)

        # db_history_index = self._load_index(self._history_storage_path)
        # 检索返回List[NodeWithScore]
        retrieved_history_nodes = self._retrieve_context_nodes(
            self._history_index, nl_query, top_k
        )
        logger.info(
            f"History nodes retrieved from index, number of nodes: {len(retrieved_history_nodes)}"
        )

        # 如有检索结果，进一步缩小db_history_list，否则返回原始list
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

        # db_history_index = self._load_index(self._history_storage_path)
        # 检索返回List[NodeWithScore]
        retrieved_history_nodes = await self._aretrieve_context_nodes(
            self._history_index, nl_query, top_k
        )
        logger.info(
            f"History nodes retrieved from index, number of nodes: {len(retrieved_history_nodes)}"
        )

        # 如有检索结果，进一步缩小db_history_list，否则返回原始list
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
        similar_entities_via_LSH: List[Dict[str, Any]],
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

        if len(similar_entities_via_LSH) > 0:
            for item in similar_entities_via_LSH:
                key = (item["table_name"], item["column_name"])
                value = item["similar_value"]
                if key not in retrieved_nodes_dict:
                    retrieved_nodes_dict[key] = [str(value)]
                else:
                    if str(value) not in retrieved_nodes_dict[key]:
                        retrieved_nodes_dict[key].append(str(value))
            logger.info(
                f"retrieved_nodes_dict: {len(retrieved_nodes_dict)},\n {retrieved_nodes_dict}"
            )
        else:
            logger.info("Empty similar_entities_via_LSH")

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

    def _load_lsh(
        self, file_path: Optional[str] = None
    ) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
        """
        Loads the LSH and MinHashes from the preprocessed files in the specified directory.

        Args:
            db_directory_path (str): The path to the database directory.

        Returns:
            Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The LSH object and the dictionary of MinHashes.

        Raises:
            Exception: If there is an error loading the LSH or MinHashes.
        """
        if not file_path:
            file_path = self._value_lsh_path

        try:
            with open(os.path.join(file_path, "lsh.pkl"), "rb") as file:
                lsh = pickle.load(file)
            with open(os.path.join(file_path, "minhashes.pkl"), "rb") as file:
                minhashes = pickle.load(file)
            logger.info(f"LSH loaded from {file_path}")
            return lsh, minhashes
        except Exception as e:
            logger.error(f"Error loading LSH: {e}")
            raise e

    def _query_lsh(
        self, keyword: str, signature_size: int = 128, n_gram: int = 3, top_n: int = 10
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Queries the LSH for similar values to the given keyword and returns the top results.

        Args:
            lsh (MinHashLSH): The LSH object.
            minhashes (Dict[str, Tuple[MinHash, str, str, str]]): The dictionary of MinHashes.
            keyword (str): The keyword to search for.
            signature_size (int, optional): The size of the MinHash signature.
            n_gram (int, optional): The n-gram size for the MinHash.
            top_n (int, optional): The number of top results to return.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary containing the top similar values.
        """
        query_minhash = create_minhash(signature_size, keyword, n_gram)
        db_value_lsh, db_value_minhashes = self._load_lsh(self._value_lsh_path)
        results = db_value_lsh.query(query_minhash)
        similarities = [
            (
                result,
                jaccard_similarity(query_minhash, db_value_minhashes[result][0]),
            )
            for result in results
        ]
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

        similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
        for result, similarity in similarities:
            table_name, column_name, value = db_value_minhashes[result][1:]
            if table_name not in similar_values_trimmed:
                similar_values_trimmed[table_name] = {}
            if column_name not in similar_values_trimmed[table_name]:
                similar_values_trimmed[table_name][column_name] = []
            similar_values_trimmed[table_name][column_name].append(value)

        return similar_values_trimmed

    def _column_value(self, string: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Splits a string into column and value parts if it contains '='.

        Args:
            string (str): The string to split.

        Returns:
            Tuple[Optional[str], Optional[str]]: The column and value parts.
        """
        if "=" in string:
            left_equal = string.find("=")
            first_part = string[:left_equal].strip()
            second_part = (
                string[left_equal + 1 :].strip()
                if len(string) > left_equal + 1
                else None
            )
            return first_part, second_part
        return None, None

    def _get_search_values(self, keywords: List[str]) -> List[str]:
        """
        Extracts values to search from the keywords.

        Args:
            keywords (List[str]): The list of keywords.

        Returns:
            List[str]: A list of values to search.
        """

        def get_substring_packet(keyword: str, substring: str) -> Dict[str, str]:
            return {"keyword": keyword, "substring": substring}

        to_search_values = []
        for keyword in keywords:
            keyword = keyword.strip()
            to_search_values.append(get_substring_packet(keyword, keyword))
            if " " in keyword:
                for i in range(len(keyword)):
                    if keyword[i] == " ":
                        first_part = keyword[:i]
                        second_part = keyword[i + 1 :]
                        to_search_values.append(
                            get_substring_packet(keyword, first_part)
                        )
                        to_search_values.append(
                            get_substring_packet(keyword, second_part)
                        )
            hint_column, hint_value = self._column_value(keyword)
            if hint_value:
                to_search_values.append(get_substring_packet(keyword, hint_value))
        to_search_values.sort(
            key=lambda x: (x["keyword"], len(x["substring"]), x["substring"]),
            reverse=True,
        )
        return to_search_values

    def _get_similar_entities_via_LSH(
        self, keywords: List[str]
    ) -> List[Dict[str, Any]]:
        substring_packets = self._get_search_values(keywords)
        similar_entities_via_LSH = []
        for packet in substring_packets:
            keyword = packet["keyword"]
            substring = packet["substring"]
            unique_similar_values = self._query_lsh(keyword=substring)
            for table_name, column_values in unique_similar_values.items():
                for column_name, values in column_values.items():
                    for value in values:
                        similar_entities_via_LSH.append(
                            {
                                "keyword": keyword,
                                "substring": substring,
                                "table_name": table_name,
                                "column_name": column_name,
                                "similar_value": value,
                            }
                        )
        return similar_entities_via_LSH
