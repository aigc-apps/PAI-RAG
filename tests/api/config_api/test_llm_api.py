import os
from typing import Generator
from fastapi.testclient import TestClient
import pytest
from httpx import Client

os.environ["SQLITE_URL"] = "sqlite+aiosqlite:///./localdata/pytest.db"
os.environ["DB_TYPE"] = "sqlite"

@pytest.fixture()
def client() -> Generator[None, None, Client]:
    from app.main import app
    with TestClient(app) as client:
        yield client

def test_list_llm(client: Client):
    response = client.get("/v1/config/llms")
    assert response.status_code == 200
    responseData = response.json()
    assert responseData["code"] == 200
    assert responseData["message"] == "获取LLM模型列表成功"

def test_create_llm(client: Client):
    create_response = client.post(
        "/v1/config/llms",
        json={
            "model_id": "gpt-test",
            "base_url": "http://localhost:8000",
            "model": "Qwen3-8B",
            "api_key": "sk-testxxxxxx",
            "temperature": 0.7,
            "context_window": 8192,
        })
    assert create_response.status_code == 200
    create_resp_json = create_response.json()
    assert create_resp_json["code"] == 200
    llm_id = create_resp_json["data"]["id"]


    get_response = client.get(f"/v1/config/llms/{llm_id}")
    assert get_response.status_code == 200
    llm_json = get_response.json()
    print(llm_json)
    assert llm_json["data"]["model_id"] == "gpt-test"
    assert llm_json["data"]["temperature"] == 0.7
    assert llm_json["data"]["context_window"] == 8192
    assert "api_key" not in llm_json["data"], "API_KEY should not be returned to frontend."

    delete_response = client.delete(f"/v1/config/llms/{llm_id}")
    assert delete_response.status_code == 200
    delete_json = delete_response.json()
    assert delete_json["message"].endswith("删除成功。")

    get_response2 = client.get(f"/v1/config/llms/{llm_id}")
    assert get_response2.status_code == 404
    deleted_model_json = get_response2.json()
    assert deleted_model_json["code"] == 404
