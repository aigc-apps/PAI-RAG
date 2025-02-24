from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from loguru import logger

from llama_index.core.schema import NodeWithScore

from pai_rag.integrations.data_analysis.text2sql.utils.info_utils import (
    count_total_columns,
    extract_subset_from_description,
)


class DBRetrieverFilter(ABC):
    """数据库信息检索后处理接口"""

    @abstractmethod
    def filter(self):
        pass


class SchemaValueFilter(DBRetrieverFilter):
    def filter(
        self,
        db_description_dict: Dict,
        retrieved_description_nodes_from_query: List[NodeWithScore],
        retrieved_description_nodes_from_hint: List[NodeWithScore],
        retrieved_value_nodes: List[NodeWithScore],
    ):
        column_nums = count_total_columns(db_description_dict)
        retrieved_description_dict = self._filter_description(
            db_description_dict,
            retrieved_description_nodes_from_query,
            retrieved_description_nodes_from_hint,
            retrieved_value_nodes,
        )
        retrieved_column_nums = count_total_columns(retrieved_description_dict)
        logger.info(
            f"Description dict filtered, number from {column_nums} to {retrieved_column_nums}."
        )
        return retrieved_description_dict

    def _filter_description(
        self,
        db_description_dict: Dict,
        retrieved_description_nodes_from_query: List[NodeWithScore],
        retrieved_description_nodes_from_hint: List[NodeWithScore],
        retrieved_value_nodes: List[NodeWithScore],
        similar_entities_via_LSH: Optional[List[Dict[str, Any]]] = None,
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
                f"After retrieved_nodes_dict from value_nodes: {len(retrieved_nodes_dict)}"
            )
        else:
            logger.info("Empty retrieved_value_nodes")

        if len(retrieved_description_nodes_from_query) > 0:
            for node in retrieved_description_nodes_from_query:
                key = (node.metadata["table_name"], node.metadata["column_name"])
                if key not in retrieved_nodes_dict:
                    retrieved_nodes_dict[key] = []
            logger.info(
                f"After retrieved_description_nodes_from_query: {len(retrieved_nodes_dict)}"
            )
        else:
            logger.info("Empty retrieved_description_nodes_from_query")

        if len(retrieved_description_nodes_from_hint) > 0:
            for node in retrieved_description_nodes_from_hint:
                key = (node.metadata["table_name"], node.metadata["column_name"])
                if key not in retrieved_nodes_dict:
                    retrieved_nodes_dict[key] = []
            logger.info(
                f"After retrieved_description_nodes_from_hint: {len(retrieved_nodes_dict)}"
            )
        else:
            logger.info("Empty retrieved_description_nodes_from_hint")

        # if similar_entities_via_LSH:
        #     for item in similar_entities_via_LSH:
        #         key = (item["table_name"], item["column_name"])
        #         value = item["similar_value"]
        #         if key not in retrieved_nodes_dict:
        #             retrieved_nodes_dict[key] = [str(value)]
        #         else:
        #             if str(value) not in retrieved_nodes_dict[key]:
        #                 retrieved_nodes_dict[key].append(str(value))
        #     logger.info(
        #         f"retrieved_nodes_dict: {len(retrieved_nodes_dict)},\n {retrieved_nodes_dict}"
        #     )
        # else:
        #     logger.info("Empty similar_entities_via_LSH")

        # 过滤db_description_dict
        filterd_db_description_dict = extract_subset_from_description(
            retrieved_nodes_dict, db_description_dict
        )

        return filterd_db_description_dict


class HistoryFilter(DBRetrieverFilter):
    def filter(self, retrieved_history_nodes: List[NodeWithScore]):
        # history_pair_nums = len(db_history_list)
        retrieved_history_list = self._filter_history(retrieved_history_nodes)

        return retrieved_history_list

    def _filter_history(self, retrieved_history_nodes: List[NodeWithScore]) -> List:
        """根据retrieve结果缩小db_history"""
        if len(retrieved_history_nodes) == 0:
            logger.info("Empty retrieved_history_nodes.")
            return []
        else:
            retrieved_db_history_list = []
            for node in retrieved_history_nodes:
                retrieved_db_history_list.append(
                    {"query": node.metadata["query"], "SQL": node.metadata["SQL"]}
                )
            return retrieved_db_history_list

    # def _filter_history(
    #     self, db_history_list: List, retrieved_history_nodes: List[NodeWithScore]
    # ) -> List:
    #     """根据retrieve结果缩小db_history"""

    #     if len(retrieved_history_nodes) == 0:
    #         logger.info("Empty retrieved_history_nodes, use original history instead.")
    #         return db_history_list

    #     else:
    #         retrieved_nodes_list = []
    #         for node in retrieved_history_nodes:
    #             retrieved_nodes_list.append({"query": node.metadata["query"]})

    #         retrieved_db_history_list = []
    #         for item in db_history_list:
    #             # 检查 item['query'] 是否在 retrieved_nodes_list 中
    #             if any(
    #                 item["query"] == filter_item["query"]
    #                 for filter_item in retrieved_nodes_list
    #             ):
    #                 retrieved_db_history_list.append(item)

    #         return retrieved_db_history_list
