import os
import pytest
from dotenv import load_dotenv
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.schema import TextNode
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.index.pai.vector_store_config import MilvusVectorStoreConfig


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

# 加载 .env 文件
load_dotenv()

# 从环境变量中读取配置
dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
milvus_host = os.getenv("MILVUS_HOST", "http://localhost:19530")
milvus_port = int(os.getenv("MILVUS_PORT", 19530))
milvus_user = os.getenv("MILVUS_USER", "")
milvus_password = os.getenv("MILVUS_PASSWORD", "")
milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME", "pairag_tests")
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

# embed_model = DashScopeEmbedding(embed_batch_size=10, api_key=dashscope_key)
# vector_store_index = PaiVectorStoreIndex(vector_store_config, embed_model=embed_model)

# vector_store_index.insert_nodes(mock_nodes)
# vector_store_index.delete_nodes(["3"])
# vector_store_index.delete_nodes(["node_3"])   # 无效
# vector_store_index.delete_nodes([])   # 无效
# vector_store_index.clear()
# res = vector_store_index._vector_store.client.list_collections()


@pytest.mark.skipif(os.getenv("PAI_RAG_MODEL_DIR") is None, reason="no model dir")
@pytest.fixture()
def setup_vector_store_index():
    embed_model = DashScopeEmbedding(embed_batch_size=10, api_key=dashscope_key)
    # 初始化 PaiVectorStoreIndex
    vector_store_index = PaiVectorStoreIndex(
        vector_store_config, embed_model=embed_model
    )
    return vector_store_index


# 测试插入节点
@pytest.mark.skipif(os.getenv("MILVUS_HOST") is None, reason="no host")
@pytest.mark.skipif(os.getenv("PAI_RAG_MODEL_DIR") is None, reason="no model dir")
def test_insert_nodes(setup_vector_store_index):
    vector_store_index = setup_vector_store_index
    vector_store_index.insert_nodes(mock_nodes)
    vector_store_index._vector_store.client.flush(collection_name="pairag_tests")
    vector_count = len(
        vector_store_index._vector_store.client.query(
            collection_name="pairag_tests", limit=5
        )
    )
    assert vector_count == len(mock_nodes)


# 测试删除节点
@pytest.mark.skipif(os.getenv("MILVUS_HOST") is None, reason="no host")
@pytest.mark.skipif(os.getenv("PAI_RAG_MODEL_DIR") is None, reason="no model dir")
def test_delete_nodes(setup_vector_store_index):
    vector_store_index = setup_vector_store_index
    # # 插入三个节点
    # vector_store_index.insert_nodes(mock_nodes)
    # 删除一个节点
    node_ids_to_delete = ["node_3"]
    vector_store_index.delete_nodes(node_ids_to_delete)
    vector_store_index.delete_nodes(["3"])  # 无效
    vector_store_index.delete_nodes([])  # 无效
    # time.sleep(1)
    vector_store_index._vector_store.client.flush(collection_name="pairag_tests")
    vector_count = len(
        vector_store_index._vector_store.client.query(
            collection_name="pairag_tests", limit=5
        )
    )

    expected_count = len(mock_nodes) - 1
    assert vector_count == expected_count


@pytest.mark.skipif(os.getenv("MILVUS_HOST") is None, reason="no host")
@pytest.mark.skipif(os.getenv("PAI_RAG_MODEL_DIR") is None, reason="no model dir")
def test_clear(setup_vector_store_index):
    vector_store_index = setup_vector_store_index

    collections_before_clear = (
        vector_store_index._vector_store.client.list_collections()
    )
    assert "pairag_tests" in collections_before_clear

    vector_store_index.clear()

    collections_after_clear = vector_store_index._vector_store.client.list_collections()
    assert "pairag_tests" not in collections_after_clear


# @pytest.mark.skipif(os.getenv("MILVUS_HOST") is None, reason="no host")
# @pytest.mark.asyncio
# async def test_adelete_nodes():
#     # 插入三个节点
#     vector_store_index.insert_nodes(mock_nodes)
#     # 删除一个节点
#     nodes_to_delete = ["node_3"]
#     await vector_store_index.adelete_nodes(nodes_to_delete)
#     vector_store_index._vector_store.client.flush(collection_name="pairag_tests")
#     vector_count = len(
#         vector_store_index._vector_store.client.query(
#             collection_name="pairag_tests", limit=5
#         )
#     )

#     expected_count = len(mock_nodes) - 1
#     assert vector_count == expected_count
