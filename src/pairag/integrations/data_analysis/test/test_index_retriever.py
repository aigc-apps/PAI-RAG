import os
from typing import List
import hashlib

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding

from pairag.integrations.data_analysis.text2sql.db_info_retriever import (
    SchemaRetriever,
)


if os.path.exists("./model_repository/bge-m3"):
    embed_model_bge = HuggingFaceEmbedding(
        model_name="./model_repository/bge-m3", embed_batch_size=20
    )
else:
    embed_model_bge = None

mock_nodes = [
    TextNode(
        text="This is mock node 0.",
        metadata={"table_name": "table0", "column_name": "column0"},
    ),
    TextNode(
        text="This is mock node 1.",
        metadata={"table_name": "table0", "column_name": "column1"},
    ),
    TextNode(
        text="This is mock node 2.",
        metadata={"table_name": "table0", "column_name": "column2"},
    ),
]


def get_nodes_with_embeddings(embed_model: BaseEmbedding, nodes: List[TextNode]):
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


mock_nodes_with_embeddings = get_nodes_with_embeddings(embed_model_bge, mock_nodes)

# 初始化检索器
mock_retriever = SchemaRetriever(
    db_name="mock_db",
    embed_model=embed_model_bge,
    similarity_top_k=1,
)

# 插入/更新nodes
mock_retriever.get_index(mock_nodes_with_embeddings)


mock_nodes_update = [
    TextNode(
        text="This is mock node 0.",
        metadata={"table_name": "table0", "column_name": "column0"},
    ),
    TextNode(
        text="This is mock node 01test.",
        metadata={"table_name": "table0", "column_name": "column1"},
    ),
    TextNode(
        text="This is mock node 2.",
        metadata={"table_name": "table0", "column_name": "column2"},
    ),
]

mock_nodes_with_embeddings = get_nodes_with_embeddings(
    embed_model_bge, mock_nodes_update
)

# 插入/更新nodes
mock_retriever.get_index(mock_nodes_with_embeddings)
# new_retriever = mock_retriever._schema_index.as_retriever()

res = mock_retriever.retrieve_nodes(query="what is mock node 2?")

print("retrieve result:", res)
