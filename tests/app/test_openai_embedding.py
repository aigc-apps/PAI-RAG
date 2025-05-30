import json
import mimetypes
import os
import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient
import time
from pairag.app.app import app


@pytest.mark.asyncio(scope="session")
async def test_embedding():
    import openai

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        client = openai.AsyncClient(
            base_url="http://test/v1", api_key="123", http_client=client
        )
        embedding_result = await client.embeddings.create(
            input="hello world",
            model="bge-m3",
        )

        assert len(embedding_result.data[0].embedding) == 1024
        assert embedding_result.data[0].index == 0

        embedding_result = await client.embeddings.create(
            input="",
            model="bge-m3",
        )

        assert len(embedding_result.data[0].embedding) == 1024
        assert embedding_result.data[0].index == 0

        embedding_result = await client.embeddings.create(
            input=["", "hi", "你在干什么"],
            model="bge-m3",
        )

        assert len(embedding_result.data[0].embedding) == 1024
        assert len(embedding_result.data[1].embedding) == 1024
        assert len(embedding_result.data[2].embedding) == 1024
        assert embedding_result.data[0].index == 0
        assert embedding_result.data[1].index == 1
        assert embedding_result.data[2].index == 2

        try:
            embedding_result = await client.embeddings.create(
                input=None,
                model="bge-m3",
            )
            raise Exception("should not reach here.")
        except Exception as e:
            print(e)
