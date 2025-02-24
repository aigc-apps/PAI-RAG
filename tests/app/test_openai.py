import json
import mimetypes
import os
import pytest
from pai_rag.app.app import app
from httpx import ASGITransport, AsyncClient

import asyncio


if (
    "DASHSCOPE_API_KEY" not in os.environ
    or os.getenv("SKIP_GPU_TESTS", "false") == "true"
):
    pytest.skip(
        allow_module_level=True,
        reason='Environment variable "DASHSCOPE_API_KEY" not set.',
    )


async def upload_file(input_files, index_name="default_index"):
    files = []
    file_obj_list = []
    if input_files:
        for file_name in input_files:
            file_obj = open(file_name, "rb")
            mimetype = mimetypes.guess_type(file_name)[0]
            files.append(("files", (os.path.basename(file_name), file_obj, mimetype)))
            file_obj_list.append(file_obj)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/upload_data", files=files, data={"index_name": index_name}
        )
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    assert task_id is not None and len(task_id) > 0

    i = 0
    task_status = "pending"
    while True and i < 20:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                f"/api/v1/get_upload_state?task_id={task_id}",
            )
        assert response.status_code == 200
        task_status = response.json()["status"]
        if task_status == "completed" or task_status == "failed":
            break

        i += 1
        await asyncio.sleep(1)

    assert task_status == "completed"

    for file in file_obj_list:
        file.close()


async def setup_app():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            "/api/v1/config",
            json={
                "llm": {
                    "source": "openai_compatible",
                    "model": "qwen-max",
                    "api_key": os.environ.get("DASHSCOPE_API_KEY", "abc"),
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                },
                "search": {
                    "source": "bing",
                    "search_api_key": os.environ.get("BING_SEARCH_KEY", "abc"),
                },
                "postprocessor": {
                    "reranker_type": "no-reranker",
                    "similarity_threshold": 0.7,
                },
                "retriever": {"retrieval_mode": "default"},
            },
        )
        assert response.status_code == 200

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/indexes")
        assert response.status_code == 200
        indexes = response.json()["indexes"]

    if "test_index" not in indexes:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/indexes/test_index",
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
            assert response.json()["msg"] == "Add index 'test_index' successfully."

    await upload_file(["tests/testdata/data/md_data/pai_document.md"])
    await upload_file(
        ["tests/testdata/paul_graham/paul_graham_essay.txt"], index_name="test_index"
    )


asyncio.run(setup_app())


@pytest.mark.asyncio(scope="session")
async def test_get_v1_path():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/v1")
    assert response.status_code == 200


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

    assert "助手" in answer
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

    assert "股价" in answer
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

    assert "股价" in answer
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

    assert "助手" in answer
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

    assert "股价" in answer
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

    assert "股价" in answer
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
                        "content": "Why does my experiment generate an empty model?",
                    }
                ],
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
            citations = chunk_data.get("citation_details", [])

    print(citations)
    assert "Machine Learning Studio" in answer
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

    print(citations)
    assert len(answer) > 0
    assert len(citations) == 0

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

    print(citations)
    assert "program" in answer
    assert len(citations) > 0
