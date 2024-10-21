import asyncio
import os
from pathlib import Path
from pai_rag.app.api.models import RagQuery
import pytest
import shutil
from pai_rag.core.rag_application import RagApplication, RagChatType
from pai_rag.core.rag_config_manager import RagConfigManager

BASE_DIR = Path(__file__).parent.parent.parent
TEST_INDEX_PATH = "localdata/teststorage"

EXPECTED_EMPTY_RESPONSE = """Empty query. Please input your question."""


@pytest.fixture(scope="module", autouse=True)
def rag_app():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfigManager.from_file(config_file).get_value()

    if os.path.isdir(TEST_INDEX_PATH):
        shutil.rmtree(TEST_INDEX_PATH)
    config.index.vector_store.persist_path = TEST_INDEX_PATH

    rag_app = RagApplication(config)
    rag_app.initialize(config)

    data_dir = os.path.join(BASE_DIR, "tests/testdata/paul_graham")
    rag_app.load_knowledge(data_dir)

    return rag_app


# Test rag query
def test_query(rag_app: RagApplication):
    query = RagQuery(question="Why did he decide to learn AI?")
    response = asyncio.run(rag_app.aquery(query, chat_type=RagChatType.RAG))
    print(response.answer)
    print(response.docs)
    assert len(response.answer) > 10 and response.answer != "Empty Response"

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery(query, chat_type=RagChatType.RAG))
    assert response.answer == EXPECTED_EMPTY_RESPONSE


# Test llm query
def test_llm(rag_app: RagApplication):
    query = RagQuery(question="What is the result of 15+22?")
    response = asyncio.run(rag_app.aquery(query, chat_type=RagChatType.LLM))
    assert "37" in response.answer

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery(query, chat_type=RagChatType.LLM))
    assert response.answer == EXPECTED_EMPTY_RESPONSE


# Test retrieval query
def test_retrieval(rag_app: RagApplication):
    retrieval_query = RagQuery(question="Why did he decide to learn AI?")
    response = asyncio.run(rag_app.aretrieve(retrieval_query))
    assert len(response.docs) > 0

    empty_query = RagQuery(question="")
    response = asyncio.run(rag_app.aretrieve(empty_query))
    assert len(response.docs) == 0


# Test agent query
def test_agent(rag_app: RagApplication):
    query = RagQuery(question="What is the result of 15+22?")
    response = asyncio.run(rag_app.aquery_agent(query))
    assert "37" in response.answer

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery_agent(query))
    assert response.answer == EXPECTED_EMPTY_RESPONSE
