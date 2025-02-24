import os
import pytest
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.index.pai.vector_store_config import MilvusVectorStoreConfig
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 设置日志记录
import logging

logging.basicConfig(level=logging.INFO)

# 初始化嵌入模型
if os.path.exists("./model_repository/bge-m3"):
    embed_model_bge = HuggingFaceEmbedding(
        model_name="./model_repository/bge-m3", embed_batch_size=20
    )
else:
    embed_model_bge = None

# 构造 mock_nodes
mock_nodes = [
    TextNode(
        text="Cat",
        ref_doc_id="doc_1",
        metadata={"node_id": "node_1", "ref_doc_id": "doc_1"},
    ),
    TextNode(
        text="Dog",
        ref_doc_id="doc_1",
        metadata={"node_id": "node_2", "ref_doc_id": "doc_1"},
    ),
    TextNode(
        text="Horse",
        ref_doc_id="doc_2",
        metadata={"node_id": "node_3", "ref_doc_id": "doc_2"},
    ),
]

# 设置 node_id
for i, node in enumerate(mock_nodes):
    node.node_id = node.metadata["node_id"]

# 从环境变量中读取配置
milvus_host = os.getenv("MILVUS_HOST", "http://localhost:19530")
milvus_port = int(os.getenv("MILVUS_PORT", 19530))
milvus_user = os.getenv("MILVUS_USER", "")
milvus_password = os.getenv("MILVUS_PASSWORD", "")
milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME", "pairagcollection")
milvus_database = os.getenv("MILVUS_DATABASE", "default")

# 配置 vector_store
vector_store_config = MilvusVectorStoreConfig(
    type="milvus",
    host=milvus_host,
    password=milvus_password,
    port=milvus_port,
    collection_name=milvus_collection_name,
    user=milvus_user,
    database=milvus_database,
)

# 初始化 PaiVectorStoreIndex
vector_store_index = PaiVectorStoreIndex(
    vector_store_config, embed_model=embed_model_bge
)


# # 测试插入节点
# @pytest.mark.skipif(os.getenv("MILVUS_HOST") is None, reason="no host")
# def test_insert_nodes():
#     vector_store_index.insert_nodes(mock_nodes)
#     vector_count = len(vector_store_index._vector_store.client.query(collection_name="pairagcollection", limit=5))
#     assert vector_count == len(mock_nodes)


# 测试删除节点
@pytest.mark.skipif(os.getenv("MILVUS_HOST") is None, reason="no host")
def test_delete_nodes():
    # 插入三个节点
    vector_store_index.insert_nodes(mock_nodes)
    # 删除一个节点
    node_to_delete = "node_3"
    vector_store_index.delete_nodes([node_to_delete])
    vector_count = len(
        vector_store_index._vector_store.client.query(
            collection_name="pairagcollection", limit=5
        )
    )
    expected_count = len(mock_nodes) - 1
    assert vector_count == expected_count
