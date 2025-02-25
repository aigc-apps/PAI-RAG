from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import json
import hashlib

from llama_index.core.schema import TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding

from pai_rag.integrations.data_analysis.text2sql.utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
    DEFAULT_DB_DESCRIPTION_NAME,
    DEFAULT_DB_HISTORY_PATH,
    DEFAULT_DB_HISTORY_NAME,
    DEFAULT_DB_VALUE_PATH,
    DEFAULT_DB_VALUE_NAME,
)


class DBInfoNode(ABC):
    """node接口，内部使用"""

    def __init__(self, db_name: str, embed_model: BaseEmbedding):
        self._db_name = db_name
        self._embed_model = embed_model

    @abstractmethod
    def create_nodes_with_embeddings(self, data: Optional[Dict | List] = None):
        pass

    @abstractmethod
    async def acreate_nodes_with_embeddings(self, data: Optional[Dict | List] = None):
        pass


class SchemaNode(DBInfoNode):
    def __init__(self, db_name: str, embed_model: BaseEmbedding):
        super().__init__(db_name, embed_model)
        db_description_path = DEFAULT_DB_DESCRIPTION_PATH
        self._db_description_path = os.path.join(
            db_description_path, f"{db_name}_{DEFAULT_DB_DESCRIPTION_NAME}"
        )

    def create_nodes_with_embeddings(self, data: Optional[Dict] = None):
        # create nodes
        schema_description_nodes = self._create_nodes_from_schema(data)
        # update nodes with embeddings
        schema_description_nodes = _get_nodes_with_embeddings(
            self._embed_model, schema_description_nodes
        )
        return schema_description_nodes

    async def acreate_nodes_with_embeddings(self, data: Optional[Dict] = None):
        # create nodes
        schema_description_nodes = self._create_nodes_from_schema(data)
        # update nodes with embeddings
        schema_description_nodes = await _aget_nodes_with_embeddings(
            self._embed_model, schema_description_nodes
        )
        return schema_description_nodes

    def _create_nodes_from_schema(self, schema_description: Optional[Dict] = None):
        if not schema_description:
            schema_description = _get_file_from_path(self._db_description_path)
        # organize description nodes format
        schema_description_nodes = self._get_nodes_from_db_description(
            schema_description
        )

        return schema_description_nodes

    def _get_nodes_from_db_description(
        self, db_description_dict: Dict
    ) -> List[TextNode]:
        schema_description_nodes = []
        for table in db_description_dict["table_info"]:
            table_desc = [
                value
                for value in [table["table_comment"], table["table_description"]]
                if value is not None
            ]
            if len(table_desc) > 0:
                table_desc = ", ".join(table_desc)
            else:
                table_desc = ""
            for column in table["column_info"]:
                column_desc = [
                    value
                    for value in [
                        column["column_comment"],
                        column["column_description"],
                    ]
                    if value is not None
                ]
                if len(column_desc) > 0:
                    column_desc = f"""{column["column_name"]}: """ + ", ".join(
                        column_desc
                    )
                else:
                    column_desc = f"""{column["column_name"]}"""

                metadata = {
                    "table_name": table["table_name"],
                    "column_name": column["column_name"],
                    "column_type": column["column_type"],
                    "table_description": table_desc,
                }
                schema_description_nodes.append(
                    TextNode(text=column_desc, metadata=metadata)
                )

        return schema_description_nodes


class HistoryNode(DBInfoNode):
    def __init__(self, db_name: str, embed_model: BaseEmbedding):
        super().__init__(db_name, embed_model)
        db_history_path = DEFAULT_DB_HISTORY_PATH
        self._db_history_path = os.path.join(
            db_history_path, f"{db_name}_{DEFAULT_DB_HISTORY_NAME}"
        )

    def create_nodes_with_embeddings(self, data: Optional[List] = None):
        # create nodes
        query_history_nodes = self._create_nodes_from_history(data)
        # update nodes with embeddings
        query_history_nodes = _get_nodes_with_embeddings(
            self._embed_model, query_history_nodes
        )
        return query_history_nodes

    async def acreate_nodes_with_embeddings(self, data: Optional[List] = None):
        # create nodes
        query_history_nodes = self._create_nodes_from_history(data)
        # update nodes with embeddings
        query_history_nodes = await _aget_nodes_with_embeddings(
            self._embed_model, query_history_nodes
        )
        return query_history_nodes

    def _create_nodes_from_history(self, query_history: Optional[List] = None):
        if not query_history:
            query_history = _get_file_from_path(self._db_history_path)
        # organize history nodes format
        query_history_nodes = self._get_nodes_from_db_history(query_history)

        return query_history_nodes

    def _get_nodes_from_db_history(self, db_query_history: List):
        query_history_nodes = []
        for item in db_query_history:
            query_history_nodes.append(
                TextNode(
                    text=item["query"],
                    metadata={"query": item["query"], "SQL": item["SQL"]},
                )
            )

        return query_history_nodes


class ValueNode(DBInfoNode):
    def __init__(self, db_name: str, embed_model: BaseEmbedding):
        super().__init__(db_name, embed_model)
        db_value_path = DEFAULT_DB_VALUE_PATH
        self._db_value_path = os.path.join(
            db_value_path, f"{db_name}_{DEFAULT_DB_VALUE_NAME}"
        )

    def create_nodes_with_embeddings(self, data: Optional[Dict] = None):
        # create nodes
        unique_value_nodes = self._create_nodes_from_value(data)
        # update nodes with embeddings
        unique_value_nodes = _get_nodes_with_embeddings(
            self._embed_model, unique_value_nodes
        )
        return unique_value_nodes

    async def acreate_nodes_with_embeddings(self, data: Optional[Dict] = None):
        # create nodes
        unique_value_nodes = self._create_nodes_from_value(data)
        # update nodes with embeddings
        unique_value_nodes = await _aget_nodes_with_embeddings(
            self._embed_model, unique_value_nodes
        )

        return unique_value_nodes

    def _create_nodes_from_value(self, unique_values: Optional[Dict] = None):
        if not unique_values:
            unique_values = _get_file_from_path(self._db_value_path)
        unique_values = self._get_nodes_from_db_values(unique_values)

        return unique_values

    def _get_nodes_from_db_values(self, unique_values: Dict):
        unique_value_nodes = []
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                for value in column_values:
                    unique_value_nodes.append(
                        TextNode(
                            text=value,
                            metadata={
                                "table_name": table_name,
                                "column_name": column_name,
                            },
                        )
                    )

        return unique_value_nodes


def _get_file_from_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        target_file = json.load(f)
        return target_file


def _get_nodes_with_embeddings(embed_model: BaseEmbedding, nodes: List[TextNode]):
    # get embeddings
    embeddings = embed_model.get_text_embedding_batch(
        [node.get_content(metadata_mode="embed") for node in nodes]
    )
    # update nodes embedding
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding
        node_info_str = node.get_metadata_str() + node.get_text()
        node.id_ = hashlib.sha256(node_info_str.encode()).hexdigest()

    return nodes


async def _aget_nodes_with_embeddings(
    embed_model: BaseEmbedding, nodes: List[TextNode]
):
    # get embeddings
    embeddings = await embed_model.aget_text_embedding_batch(
        [node.get_content(metadata_mode="embed") for node in nodes]
    )
    # update nodes embedding
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding
        node_info_str = node.get_metadata_str() + node.get_text()
        node.id_ = hashlib.sha256(node_info_str.encode()).hexdigest()

    return nodes
