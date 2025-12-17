from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from loguru import logger
import asyncio


async def cleanup_vector_store_async(vector_store: BasePydanticVectorStore):
    try:
        if hasattr(vector_store, "close"):
            if asyncio.iscoroutinefunction(vector_store.close):
                await vector_store.close()
            else:
                vector_store.close()
    except Exception as e:
        logger.warning(f"Error closing vector store: {e}")


def cleanup_vector_store(vector_store: BasePydanticVectorStore):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(cleanup_vector_store_async(vector_store))
        return

    if loop.is_running():
        loop.create_task(cleanup_vector_store_async(vector_store))
    else:
        asyncio.run(cleanup_vector_store_async(vector_store))


def is_docid_filter_supported(vector_store: BasePydanticVectorStore) -> bool:
    """
    Check if the vector store supports filtering by docid.

    Args:
        vector_store (BasePydanticVectorStore): The vector store to check.

    Returns:
        bool: True if the vector store supports filtering by docid, False otherwise.
    """
    if isinstance(vector_store, PGVectorStore):
        return False
    else:
        return True
