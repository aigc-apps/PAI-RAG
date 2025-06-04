import json
import mimetypes
import os
import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient
import time
from pairag.app.app import app

DEFAULT_GUARDRAIL_RESPONSE = "抱歉，无法处理这个请求。"
DEFAULT_EMPTY_RESPONSE = "看起来你发了一条空白消息，有什么能帮到你的吗？"
DEFAULT_ERROR_RESPONSE = "抱歉，系统出错，暂时无法处理这个请求。"


def upload_file(input_files, index_name="default"):
    files = []
    file_obj_list = []
    file_names_added = []
    if input_files:
        for file_name in input_files:
            file_obj = open(file_name, "rb")
            mimetype = mimetypes.guess_type(file_name)[0]
            files.append(("files", (os.path.basename(file_name), file_obj, mimetype)))
            file_obj_list.append(file_obj)
            file_names_added.append(os.path.basename(file_name))

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/knowledgebases/{index_name}/files", files=files
        )
        assert response.status_code == 200

        i = 0
        task_status = "pending"

        for file_name in file_names_added:
            while True and i < 40:
                response = client.get(
                    f"/api/v1/knowledgebases/{index_name}/files/{file_name}",
                )
                if response.status_code != 200:
                    time.sleep(1)
                    continue

                assert response.status_code == 200
                task_status = response.json()["status"]
                if task_status == "done" or task_status == "failed":
                    print(response.json())
                    break

                i += 1
                time.sleep(1)

        assert task_status == "done"

    for file in file_obj_list:
        file.close()


def setup_app():
    with TestClient(app) as client:
        response = client.patch(
            "/api/v1/config",
            json={
                "llms": [
                    {
                        "source": "openai_compatible",
                        "model": "qwen-max",
                        "api_key": os.environ.get("DASHSCOPE_API_KEY", "abc"),
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    }
                ],
                "trace": [
                    {
                        "endpoint": os.environ.get(
                            "TRACE_ENDPOINT",
                            "http://tracing-analysis-dc-hz.aliyuncs.com:8090",
                        ),
                        "token": os.environ.get("TRACE_TOKEN", "abc"),
                        "service_name": os.environ.get(
                            "TRACE_SERVICE_NAME", "pai_rag_test_gpu4"
                        ),
                        "enabled": True,
                    }
                ],
                "search": {
                    "source": "bing",
                    "search_api_key": os.environ.get("BING_SEARCH_KEY", "abc"),
                },
                "postprocessor": {
                    "reranker_type": "no-reranker",
                },
                "retriever": {"retrieval_mode": "default"},
            },
        )
        assert response.status_code == 200

    with TestClient(app) as client:
        response = client.get("/api/v1/knowledgebases")
        assert response.status_code == 200
        indexes = response.json()["knowledgebases"]

    if "test_index" not in indexes:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/knowledgebases/test_index",
                json={
                    "index_name": "test_index",
                    "embedding_config": {
                        "source": "dashscope",
                        "api_key": os.environ.get("DASHSCOPE_API_KEY", "abc"),
                    },
                    "vector_store_config": {
                        "type": "faiss",
                        "persist_path": "localdata/storage/test_index",
                    },
                },
            )

            assert response.status_code == 200
            assert (
                response.json()["msg"] == "Add knowledgebase 'test_index' successfully."
            )

    upload_file(["tests/testdata/pai_document.md"])
    upload_file(
        ["tests/testdata/paul_graham/paul_graham_essay.txt"], index_name="test_index"
    )


setup_app()


@pytest.mark.asyncio(scope="session")
async def test_openai_chat():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "中国首都在哪里"}],
                "search_web": False,
                "stream": False,
            },
        )
    assert response.status_code == 200

    answer = response.json()["choices"][0]["message"]["content"]

    assert "北京" in answer


@pytest.mark.asyncio(scope="session")
async def test_openai_chat_stream():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "中国首都在哪里"}],
                "search_web": False,
                "stream": True,
            },
        )
    assert response.status_code == 200

    answer = ""
    for chunk in response.iter_lines():
        if chunk.startswith("data:"):
            chunk = chunk[5:]
            chunk_data = json.loads(chunk)
            delta = chunk_data["choices"][0]["delta"]["content"]
            answer += delta

    assert "北京" in answer


@pytest.mark.asyncio(scope="session")
async def test_openai_websearch():
    # 普通问题，不会search
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "你是谁"}],
                "search_web": True,
                "stream": False,
                "return_reference": True,
            },
        )
    assert response.status_code == 200

    answer = response.json()["choices"][0]["message"]["content"]

    assert len(answer) > 0
    assert len(response.json()["citations"]) == 0

    # search并返回reference
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "最新的阿里巴巴股价"}],
                "search_web": True,
                "stream": False,
                "return_reference": True,
            },
        )
    assert response.status_code == 200

    answer = response.json()["choices"][0]["message"]["content"]

    assert (
        answer != DEFAULT_GUARDRAIL_RESPONSE
        and answer != DEFAULT_EMPTY_RESPONSE
        and answer != DEFAULT_ERROR_RESPONSE
    )
    assert len(response.json()["citations"]) > 0

    # 不返回reference
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "最新的阿里巴巴股价"}],
                "search_web": True,
                "stream": False,
                "return_reference": False,
            },
        )
    assert response.status_code == 200

    answer = response.json()["choices"][0]["message"]["content"]

    assert (
        answer != DEFAULT_GUARDRAIL_RESPONSE
        and answer != DEFAULT_EMPTY_RESPONSE
        and answer != DEFAULT_ERROR_RESPONSE
    )
    assert len(response.json()["citations"]) == 0


@pytest.mark.asyncio(scope="session")
async def test_openai_websearch_stream():
    # 普通问题，不会search
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "你是谁"}],
                "search_web": True,
                "stream": True,
                "return_reference": True,
            },
        )
    assert response.status_code == 200

    answer = ""
    citations = []
    for chunk in response.iter_lines():
        if chunk.startswith("data:"):
            chunk = chunk[5:]
            chunk_data = json.loads(chunk)
            delta = chunk_data["choices"][0]["delta"]["content"]
            answer += delta
            citations = chunk_data.get("citations", [])

    assert len(answer) > 0
    assert len(citations) == 0

    # search并返回引用
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "最新的阿里巴巴股价"}],
                "search_web": True,
                "stream": True,
                "return_reference": True,
            },
        )
    assert response.status_code == 200

    answer = ""
    citations = []
    for chunk in response.iter_lines():
        if chunk.startswith("data:"):
            chunk = chunk[5:]
            chunk_data = json.loads(chunk)
            delta = chunk_data["choices"][0]["delta"]["content"]
            answer += delta
            citations = chunk_data.get("citations", [])

    assert (
        answer != DEFAULT_GUARDRAIL_RESPONSE
        and answer != DEFAULT_EMPTY_RESPONSE
        and answer != DEFAULT_ERROR_RESPONSE
    )
    assert len(citations) > 0

    # 不返回引用
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "最新的阿里巴巴股价"}],
                "search_web": True,
                "stream": True,
                "return_reference": False,
            },
        )
    assert response.status_code == 200

    answer = ""
    citations = []
    for chunk in response.iter_lines():
        if chunk.startswith("data:"):
            chunk = chunk[5:]
            chunk_data = json.loads(chunk)
            delta = chunk_data["choices"][0]["delta"]["content"]
            answer += delta
            citations = chunk_data.get("citations", [])

    assert (
        answer != DEFAULT_GUARDRAIL_RESPONSE
        and answer != DEFAULT_EMPTY_RESPONSE
        and answer != DEFAULT_ERROR_RESPONSE
    )
    assert len(citations) == 0


@pytest.mark.asyncio(scope="session")
async def test_rag_chat():
    # 不相关问题
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "Where do you recommend for a good trip to China?",
                    }
                ],
                "stream": True,
                "chat_knowledgebase": True,
                "return_reference": True,
            },
        )
    assert response.status_code == 200
    answer = ""
    citations = []
    for chunk in response.iter_lines():
        if chunk.startswith("data:"):
            chunk = chunk[5:]
            chunk_data = json.loads(chunk)
            delta = chunk_data["choices"][0]["delta"]["content"]
            answer += delta
            citations = chunk_data.get("citations", [])

    assert len(answer) > 0
    assert len(citations) == 2

    # 相关问题
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "为什么会生成空模型?",
                    }
                ],
                "stream": True,
                "chat_knowledgebase": True,
                "return_reference": True,
            },
        )
    assert response.status_code == 200
    answer = ""
    citations = []
    for chunk in response.iter_lines():
        if chunk.startswith("data:"):
            chunk = chunk[5:]
            chunk_data = json.loads(chunk)
            delta = chunk_data["choices"][0]["delta"]["content"]
            answer += delta
            citations = chunk_data.get("citation_details", [])

    assert "空" in answer
    assert len(citations) > 0

    # 使用另一个index提问
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "Why does my experiment generate an empty model?",
                    }
                ],
                "stream": True,
                "chat_knowledgebase": True,
                "return_reference": True,
                "index_name": "test_index",  # change to test_index
            },
        )
    assert response.status_code == 200
    answer = ""
    citations = []
    for chunk in response.iter_lines():
        if chunk.startswith("data:"):
            chunk = chunk[5:]
            chunk_data = json.loads(chunk)
            delta = chunk_data["choices"][0]["delta"]["content"]
            answer += delta
            citations = chunk_data.get("citation_details", [])

    assert len(answer) > 0
    assert len(citations) == 2

    # 使用相关的index名字
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": "What did the author do growing up?"}
                ],
                "stream": True,
                "chat_knowledgebase": True,
                "return_reference": True,
                "index_name": "test_index",  # change to test_index
            },
        )
    assert response.status_code == 200

    answer = ""
    citations = []
    for chunk in response.iter_lines():
        if chunk.startswith("data:"):
            chunk = chunk[5:]
            chunk_data = json.loads(chunk)
            delta = chunk_data["choices"][0]["delta"]["content"]
            answer += delta
            citations = chunk_data.get("citation_details", [])

    # assert "program" in answer
    assert len(citations) > 0
