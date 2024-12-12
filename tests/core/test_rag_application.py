import asyncio
import os
from pathlib import Path
import pytest
import shutil

BASE_DIR = Path(__file__).parent.parent.parent
TEST_INDEX_PATH = "localdata/teststorage"

EXPECTED_EMPTY_RESPONSE = """Empty query. Please input your question."""


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_GPU_TESTS", "false") == "true",
    reason="Need to execute in a CUDA environment.",
)


@pytest.fixture(scope="module", autouse=True)
def rag_app():
    from pai_rag.core.rag_application import RagApplication
    from pai_rag.core.rag_config_manager import RagConfigManager

    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfigManager.from_file(config_file).get_value()

    if os.path.isdir(TEST_INDEX_PATH):
        shutil.rmtree(TEST_INDEX_PATH)
    config.index.vector_store.persist_path = TEST_INDEX_PATH

    rag_app = RagApplication(config)

    data_dir = os.path.join(BASE_DIR, "tests/testdata/paul_graham")
    rag_app.load_knowledge(data_dir)

    return rag_app


# Test rag query
def test_query(rag_app):
    from pai_rag.app.api.models import RagQuery
    from pai_rag.core.rag_application import RagChatType

    query = RagQuery(question="Why did he decide to learn AI?")
    response = asyncio.run(rag_app.aquery(query, chat_type=RagChatType.RAG))
    assert len(response.answer) > 10 and response.answer != "Empty Response"

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery(query, chat_type=RagChatType.RAG))
    assert response.answer == EXPECTED_EMPTY_RESPONSE


# Test llm query
def test_llm(rag_app):
    from pai_rag.app.api.models import RagQuery
    from pai_rag.core.rag_application import RagChatType

    query = RagQuery(question="What is the result of 15+22?")
    response = asyncio.run(rag_app.aquery(query, chat_type=RagChatType.LLM))
    assert "37" in response.answer

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery(query, chat_type=RagChatType.LLM))
    assert response.answer == EXPECTED_EMPTY_RESPONSE


# Test retrieval query
def test_retrieval(rag_app):
    from pai_rag.app.api.models import RagQuery

    retrieval_query = RagQuery(question="Why did he decide to learn AI?")
    response = asyncio.run(rag_app.aretrieve(retrieval_query))
    assert len(response.docs) > 0

    empty_query = RagQuery(question="")
    response = asyncio.run(rag_app.aretrieve(empty_query))
    assert len(response.docs) == 0


# Test agent query
def test_agent(rag_app):
    from pai_rag.app.api.models import RagQuery

    query = RagQuery(question="What is the result of 15+22?")
    response = asyncio.run(rag_app.aquery_agent(query))
    assert "37" in response.answer

    query = RagQuery(question="")
    response = asyncio.run(rag_app.aquery_agent(query))
    assert response.answer == EXPECTED_EMPTY_RESPONSE
