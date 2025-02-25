import os
import pytest
from dotenv import load_dotenv
import asyncio
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.index.pai.vector_store_config import ElasticSearchVectorStoreConfig


from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# # 假设你有一个文本字符串 text_str
# text_str1 = "dog"
# text_str2 = "cat"
# text_str3 = "horse"

# # 创建一个带有 doc_id 的 Document 对象
# doc1 = Document(text=text_str1, id_="doc_id_1")
# doc2 = Document(text=text_str2, id_="doc_id_2")
# doc3 = Document(text=text_str3, id_="doc_id_3")


# # 初始化 SentenceSplitter
# parser = SentenceSplitter()
# # 使用 parser 从文档中获取节点
# mock_nodes = parser.get_nodes_from_documents([doc1, doc2, doc3])

# 构造 mock_nodes
mock_nodes = [
    TextNode(
        text="Cat",
        ref_doc_id="ref_doc_1",
        doc_id="doc_1",
        metadata={"node_id": "node_1", "ref_doc_id": "ref_1","doc_id":"doc_1"},
    ),
    TextNode(
        text="Dog",
        ref_doc_id="ref_doc_1",
        doc_id="doc_2",
        metadata={"node_id": "node_2", "ref_doc_id": "ref_2","doc_id":"doc_2"},
    ),
    TextNode(
        text="Horse",
        ref_doc_id="ref_doc_2",
        doc_id="doc_3",
        metadata={"node_id": "node_3", "ref_doc_id": "ref_3", "doc_id":"doc_3"},
    ),
]

# 设置 node_id
for i, node in enumerate(mock_nodes):
    node.node_id = node.metadata["node_id"]

# 加载 .env 文件
load_dotenv()

# 从环境变量中读取配置
dashscope_key = os.getenv("DASHSCOPE_API_KEY","")
es_host = os.getenv("es_host", "")
es_port = "9200"
es_user = os.getenv("es_username", "")
es_password = os.getenv("es_password", "")


# 配置 vector_store
vector_store_config = ElasticSearchVectorStoreConfig(
    type="elasticsearch",
    es_password=es_password,
    es_url=f"http://{es_host}:{es_port}",
    es_user=es_user,
    es_index="pairag_test",
)

embed_model = DashScopeEmbedding(embed_batch_size=10, api_key=dashscope_key)
# 初始化 PaiVectorStoreIndex
vector_store_index = PaiVectorStoreIndex(vector_store_config, embed_model=embed_model)

# vector_store_index.insert_nodes(mock_nodes)
# node_ids_to_delete = ["node_3"]
# vector_store_index.delete_nodes(node_ids_to_delete)

@pytest.mark.skipif(os.getenv("es_host") is None, reason="no host")
def test_insert_nodes():
    # 插入三个节点
    vector_store_index.insert_nodes(mock_nodes)
    vector_count = asyncio.run(vector_store_index._vector_store.client.count(index="pairag_test"))["count"]
    expected_count = len(mock_nodes)
    assert vector_count == expected_count


# 测试删除节点
@pytest.mark.skipif(os.getenv("es_host") is None, reason="no host")
def test_delete_nodes():
    # # 插入三个节点
    # vector_store_index.insert_nodes(mock_nodes)
    # 删除一个节点
    node_ids_to_delete = ["node_3"]
    vector_store_index.delete_nodes(node_ids_to_delete)
    vector_count = asyncio.run(vector_store_index._vector_store.client.count(index="pairag_test"))["count"]
    expected_count = len(mock_nodes) - 1
    assert vector_count == expected_count

