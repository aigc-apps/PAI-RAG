### Vector db configuration API ###

import traceback

from fastapi import APIRouter, Depends
from rag.vector_store.vector_connection import cleanup_vector_store_async
from service.factory.vectordb_factory import create_vector_store

from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.vectordb import VectorDbConfig
from common.chat.response_model import ResponseModel, success_response
from db.db_context import get_db_session
from service.knowledgebase.vectordb_service import VectordbService
from service.injection import get_vectordb_service, get_tenant_id
from api.api_exception import ApiException
from loguru import logger


vectordb_router = APIRouter()


async def _cleanup_cached_vector_stores():
    """
    Clean up all cached vector stores in kb_cache to ensure connections are properly closed.
    This should be called before updating vector db config to prevent connection leaks.
    """
    try:
        from tools.knowledgebase.knowledgebase_tool import kb_cache

        cache_size = kb_cache.size()
        if cache_size > 0:
            logger.info(
                f"Clearing {cache_size} cached vector stores due to vector db config change."
            )
            kb_cache.clear()
        logger.info("Cleared all cached vector stores.")
    except Exception as e:
        logger.warning(f"Error cleaning up cached vector stores: {e}")


@vectordb_router.post("", response_model=ResponseModel[VectorDbConfig])
async def add_vector_db_config(
    new_config: VectorDbConfig,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    vectordb_service: VectordbService = Depends(get_vectordb_service),
):
    try:
        # Ensure type is set in config
        new_config.config["type"] = new_config.type

        # Create or update config using service
        existing_vector_config = await vectordb_service.create_or_update_vectordb_config(
            config_data=new_config,
            tenant_id=tenant_id
        )
        await session.commit()
        await session.refresh(existing_vector_config)

        # 在更新配置前，清理所有缓存的向量存储，确保连接被正确关闭
        await _cleanup_cached_vector_stores()

        return success_response(
            data=existing_vector_config, message="Update vector database success."
        )
    except ValueError as e:
        logger.error(f"Failed to add vector db config: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to add vector db config: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"Update vector database failed: {e}")


@vectordb_router.get("", response_model=ResponseModel[VectorDbConfig])
async def get_vector_config(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    vectordb_service: VectordbService = Depends(get_vectordb_service),
):
    try:
        vector_config = await vectordb_service.get_vectordb_config(tenant_id=tenant_id)
        return success_response(data=vector_config, message="Get vector database success.")
    except Exception as e:
        logger.error(f"Failed to get vector config: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Get vector database failed: {e}")


@vectordb_router.post("/connection_test", response_model=ResponseModel[dict])
async def connection_test(
    test_config: VectorDbConfig,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    vectordb_service: VectordbService = Depends(get_vectordb_service),
):
    # Prepare test config by filling in encrypted fields
    test_config = await vectordb_service.prepare_test_config(test_config=test_config, tenant_id=tenant_id)


    vector_store = None
    try:
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores import VectorStoreQuery
        import numpy as np
        vector_store = create_vector_store(
            kb_id="connectiontest",
            dimension=1024,
            vector_config=test_config,
            table_name="connectiontest",
        )
        embedding = list(np.random.rand(1024)) # convert to list for JSON serializable (HologresVectorStore requirement)
        node = TextNode(
            text="This is a test",
            id_="test",
            embedding=embedding,
            metadata={},
        )
        ids = await vector_store.async_add([node])
        assert len(ids) == 1, "Insert into vector store failed."

        vector_query = VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=3,
            query_str="test",
            mode="default",
            alpha=0.5,
        )
        results = await vector_store.aquery(vector_query)
        assert len(results.nodes) >= 1, "Query vector store failed."

        return success_response(data={}, message="Connection test success.")
    except Exception as e:
        logger.error(f"Connection test failed. \nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"Connection test failed: {e}")
    finally:
        # 确保无论成功还是失败都清理连接，避免连接泄漏
        if vector_store is not None:
            await cleanup_vector_store_async(vector_store=vector_store)
