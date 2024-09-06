import asyncio
import os
from pathlib import Path
from pai_rag.app.api.models import RagQuery
import pytest
import shutil
from pai_rag.core.rag_application import RagApplication
from pai_rag.core.rag_configuration import RagConfiguration

BASE_DIR = Path(__file__).parent.parent.parent
TEST_INDEX_PATH = "localdata/teststorage"

EXPECTED_EMPTY_RESPONSE = """Empty query. Please input your question."""


@pytest.fixture
def rag_app():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfiguration.from_file(config_file).get_value()

    if os.path.isdir(TEST_INDEX_PATH):
        shutil.rmtree(TEST_INDEX_PATH)
    config.rag.index.update({"persist_path": TEST_INDEX_PATH})

    rag_app = RagApplication()
    rag_app.initialize(config)

    data_dir = os.path.join(BASE_DIR, "tests/testdata/paul_graham")
    rag_app.load_knowledge(data_dir)

    return rag_app


# Test rag query
def test_query(rag_app: RagApplication):
    query = RagQuery(question="Why did he decide to learn AI?")
    response = asyncio.run(rag_app.aquery(query))
    assert len(response.answer) > 10

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery(query))
    assert response.answer == EXPECTED_EMPTY_RESPONSE


# Test llm query
def test_llm(rag_app: RagApplication):
    query = RagQuery(question="What is the result of 15+22?")
    response = asyncio.run(rag_app.aquery_llm(query))
    assert "37" in response.answer

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery_llm(query))
    assert response.answer == EXPECTED_EMPTY_RESPONSE


# Test retrieval query
def test_retrieval(rag_app: RagApplication):
    retrieval_query = RagQuery(question="Why did he decide to learn AI?")
    response = asyncio.run(rag_app.aquery_retrieval(retrieval_query))
    assert len(response.docs) > 0

    empty_query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery_retrieval(empty_query))
    assert len(response.docs) == 0


# Test agent query
def test_agent(rag_app: RagApplication):
    query = RagQuery(question="What is the result of 15+22?")
    response = asyncio.run(rag_app.aquery_agent(query))
    assert "37" in response.answer

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery_agent(query))
    assert response.answer == EXPECTED_EMPTY_RESPONSE


# async def test_batch_evaluate_retrieval_and_response(rag_app: RagApplication):
#     _, eval_res_avg = await rag_app.aevaluate_retrieval_and_response(type="all")
#     print('eval_res_avg', eval_res_avg)


def test_load_evaluation_qa_dataset(rag_app: RagApplication):
    qa_dataset = asyncio.run(rag_app.aload_evaluation_qa_dataset())
    print(qa_dataset)
