import os
import pytest
from typing import List
from llama_index.core.schema import TextNode
from elasticsearch import Elasticsearch
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pai_rag.integrations.vector_stores.elasticsearch.my_elasticsearch import (
    MyElasticsearchStore,
)


@pytest.fixture()
def es_connection():
    es = None
    es_username = os.getenv("es_username")
    es_password = os.getenv("es_password")
    es_host = os.getenv("es_host")
    es = Elasticsearch(
        [{"host": es_host, "port": 9200, "scheme": "http"}],
        basic_auth=(str(es_username), str(es_password)),
    )

    if es.ping():
        print("es_server connected")
    else:
        print("es_server not connected")
    return es


@pytest.mark.skipif(os.getenv("es_username") is None, reason="no username")
def test_es_analyzer(es_connection: Elasticsearch):
    """
    index: es_test1 uses the default tokenizer (standard)
    index: es_test2 uses the specified tokenizer (ik_smart) with customized new_word dict and stopword dict
    """
    es = es_connection
    text = "健康码和一键助眠的功能"

    res1 = es.indices.analyze(index="es_test1", body={"text": text})
    res1_token = [i["token"] for i in res1["tokens"]]
    assert res1_token == ["健", "康", "码", "和", "一", "键", "助", "眠", "的", "功", "能"]

    res2 = es.indices.analyze(index="es_test2", body={"text": text})
    res2_token = [i["token"] for i in res2["tokens"]]
    assert res2_token == ["健康码", "和", "一键助眠", "功能"]


@pytest.mark.skipif(os.getenv("es_username") is None, reason="no username")
def test_es_search(es_connection: Elasticsearch):
    es = es_connection
    query_match_text = "一键助眠"

    res1 = es.search(
        index="es_test1", query={"match": {"content": {"query": query_match_text}}}
    )
    assert res1["hits"]["total"]["value"] == 39

    res2 = es.search(
        index="es_test2", query={"match": {"content": {"query": query_match_text}}}
    )
    assert res2["hits"]["total"]["value"] == 6


@pytest.fixture()
def es_store():
    index_name = "py_test"
    es_host = os.getenv("es_host")
    es_port = "9200"
    es_user = os.getenv("es_username")
    es_password = os.getenv("es_password")

    es_cloud = MyElasticsearchStore(
        index_name=index_name,
        es_url=f"http://{es_host}:{es_port}",
        es_user=es_user,
        es_password=es_password,
    )

    return es_cloud


@pytest.mark.skipif(os.getenv("es_username") is None, reason="no username")
def test_es_store_add_query(es_store: MyElasticsearchStore):
    es_cloud = es_store
    text1 = "健康码是疫情期间的一种发明"
    text2 = "健康非常重要"
    node1 = TextNode(text=text1, embedding=[0.1] * 1536, id_="n1")
    node2 = TextNode(text=text2, embedding=[0.2] * 1536, id_="n2")
    nodes = [node1, node2]

    res_add = es_cloud.add(nodes)
    assert isinstance(res_add, List) is True

    vsq = VectorStoreQuery(query_embedding=[0.15] * 1536, query_str="健康码是什么")
    res_query = es_cloud.query(vsq)

    es_cloud.close()

    assert isinstance(res_query, VectorStoreQueryResult) is True
    assert res_query.nodes[0].text == text1
