### Vector db configuration API ###

import traceback
from common.knowledgebase.types import SUPPORTED_VECTOR_DB_TYPES
from config.providers.vectordb_provider import DEFAULT_VECTOR_ID, create_vector_db_connection_from_dict, vectordb_provider
from config.utils.vectordb import create_vector_db_connection_from_env
from fastapi import APIRouter, Depends
from rag.vector_store.vector_connection import create_vector_store, cleanup_vector_store_async
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.vectordb import (
    VectorDbConfig,
)
from api.response_model import ResponseModel, error_response, success_response
from db.db_context import get_session
from config.providers.config_change_manager import config_change_manager
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
    session: AsyncSession = Depends(get_session),
):
    if new_config.type not in SUPPORTED_VECTOR_DB_TYPES:
        return error_response(code=400, message=f"不支持的搜索引擎类型，当前仅支持{','.join(SUPPORTED_VECTOR_DB_TYPES)}。")

    new_config.config["type"] = new_config.type
    existing_vector_config = await session.get(VectorDbConfig, DEFAULT_VECTOR_ID)
    if existing_vector_config is None:
        logger.info(f"Adding new vectordb config for type {new_config.type}")
        if not new_config.config.get("password"):
            env_connection = create_vector_db_connection_from_env()
            new_config.config["encrypted_password"] = env_connection.model_dump().get("encrypted_password")
        if not new_config.config.get("sk"):
            env_connection = create_vector_db_connection_from_env()
            new_config.config["encrypted_sk"] = env_connection.model_dump().get("encrypted_sk")

        existing_vector_config = VectorDbConfig(
            id=DEFAULT_VECTOR_ID,
            type=new_config.type,
            config=create_vector_db_connection_from_dict(new_config.config).model_dump(),
        )
    else:
        existing_vector_config.type = new_config.type
        if not new_config.config.get("password"):
            new_config.config["encrypted_password"] = existing_vector_config.config.get(
                "encrypted_password"
            )
        if not new_config.config.get("sk"):
            new_config.config["encrypted_sk"] = existing_vector_config.config.get(
                "encrypted_sk"
            )

        existing_vector_config.config = create_vector_db_connection_from_dict(new_config.config).model_dump()

    session.add(existing_vector_config)
    try:
        await session.commit()
        await session.refresh(existing_vector_config)

        # 在更新配置前，清理所有缓存的向量存储，确保连接被正确关闭
        await _cleanup_cached_vector_stores()

        vectordb_provider.update(existing_vector_config)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.VECTORDB,
            source_id=existing_vector_config.id,
            event_type=ChangeEventType.UPDATE,
        )

        return success_response(data=existing_vector_config, message="更新向量数据库成功")
    except Exception as e:
        logger.error(f"Failed to add search config: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=500, message=f"更新向量数据库失败: {e}")


@vectordb_router.get("", response_model=ResponseModel[VectorDbConfig])
async def get_vector_config(
    session: AsyncSession = Depends(get_session),
):
    vector_config = await session.get(VectorDbConfig, DEFAULT_VECTOR_ID)
    if vector_config is None:
        connection = create_vector_db_connection_from_env()
        vector_config = VectorDbConfig(
            id=DEFAULT_VECTOR_ID,
            type=connection.type.value,
            config=connection.model_dump(),
        )


    return success_response(
        data=vector_config,
        message="查询向量数据库成功"
    )


@vectordb_router.post("/connection_test", response_model=ResponseModel[dict])
async def connection_test(
    test_config: VectorDbConfig,
    session: AsyncSession = Depends(get_session),
):
    if test_config.type != "local":
        if not test_config.config.get("password"):
            existing_vector_config = await session.get(VectorDbConfig, DEFAULT_VECTOR_ID)
            if existing_vector_config is not None:
                test_config.config["encrypted_password"] = existing_vector_config.config.get("encrypted_password")
            else:
                env_connection = create_vector_db_connection_from_env()
                test_config.config["encrypted_password"] = env_connection.model_dump().get("encrypted_password")

        if not test_config.config.get("sk"):
            existing_vector_config = await session.get(VectorDbConfig, DEFAULT_VECTOR_ID)
            if existing_vector_config is not None:
                test_config.config["encrypted_sk"] = existing_vector_config.config.get("encrypted_sk")
            else:
                env_connection = create_vector_db_connection_from_env()
                test_config.config["encrypted_sk"] = env_connection.model_dump().get("encrypted_sk")


    vector_connection = create_vector_db_connection_from_dict(test_config.config)
    vector_store = None
    try:
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores import VectorStoreQuery
        import numpy as np
        vector_store = create_vector_store(
            "connectiontest", 1024, vector_db_connection=vector_connection,
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

        logger.info("Test vector store connection success.")
        return success_response(data={}, message="测试成功。")
    except Exception as e:
        logger.error(f"测试向量库连接失败: {traceback.format_exc()}")
        return error_response(
            code=400, message=f"测试向量库连接失败: {e}"
        )
    finally:
        # 确保无论成功还是失败都清理连接，避免连接泄漏
        if vector_store is not None:
            await cleanup_vector_store_async(vector_store)
