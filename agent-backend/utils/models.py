import httpx
from utils.constants import BACKEND_PORT
from loguru import logger


async def fetch_llm(model_id: str):
    try:
        port = BACKEND_PORT
        logger.info(f"/api/chat BACKEND_PORT {port}")

        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:{port}/api/configs")
            response.raise_for_status()

            config_data = response.json()

            for item in config_data.get("llm_config", []):
                if item["id"] == model_id:
                    return item

    except httpx.HTTPError as fetch_error:
        logger.exception("Failed to fetch MCP server configurations")
        raise fetch_error

    return None
