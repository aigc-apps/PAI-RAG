import pytest
from httpx import ASGITransport, AsyncClient
from app.app import app


pytest.skip("Skip.", allow_module_level=True)


@pytest.mark.asyncio(scope="session")
async def test_intent():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat/intent",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "今天有哪些国际新闻"}]}
                ],
                "search_web": True,
                "chat_news": True,
            },
        )
        assert response.status_code == 200

        result = response.json()
        assert result["intent"] == "chat_news"
