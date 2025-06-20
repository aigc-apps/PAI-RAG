### Knowledgebase configuration API ###

import asyncio
from typing import List
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models.knowledgebase.knowledgebase import (
    ChunkConfig,
    KnowledgebaseEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from pairag.db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.mcp_tool_provider import mcp_provider
from pairag.mcp.providers.embedding_provider import embedding_provider
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider
from pairag.api.response_model import ResponseModel, success_response, error_response
from loguru import logger

knowledgebase_router = APIRouter()


@knowledgebase_router.post("", response_model=ResponseModel[KnowledgebaseEntity])
async def create_knowledgebase(
    kb: KnowledgebaseCreate, session: AsyncSession = Depends(get_session)
):
    try:
        assert kb.embedding_model, "需要提供Embedding模型才能创建知识库。"
        # 验证embedding合法
        _ = embedding_provider.get_embedding_config(kb.embedding_model)

        kb.doc_num = 0
        kb.chunk_num = 0
        kb.chunk_config = (kb.chunk_config or ChunkConfig()).model_dump()
        kb.retrieval_config = (kb.retrieval_config or RetrievalConfig()).model_dump()

        knowledgebase = KnowledgebaseEntity.model_validate(kb)
        session.add(knowledgebase)
        await session.commit()
        await session.refresh(knowledgebase)
        asyncio.create_task(knowledgebase_provider.refresh())
        return success_response(data=knowledgebase, message="知识库创建成功。")

    except IntegrityError as e:
        # TODO: 这里有bug,logger.exception没有打印错误调用栈
        logger.exception("创建知识库失败。")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return JSONResponse(
                content=error_response(code=400, message="创建知识库失败: 知识库名称已存在。"),
                status_code=400,
            )
        else:
            return JSONResponse(
                content=error_response(code=400, message=f"创建知识库失败: {e}."),
                status_code=400,
            )
    except Exception as e:
        logger.exception("创建知识库失败")
        await session.rollback()
        return JSONResponse(
            content=error_response(code=400, message=f"创建知识库失败: {e}."),
            status_code=400,
        )


@knowledgebase_router.get("", response_model=ResponseModel[List[KnowledgebaseEntity]])
async def list_knowledgebases(
    session: AsyncSession = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    knowledgebase_results = await session.exec(
        select(KnowledgebaseEntity).offset(offset).limit(limit)
    )
    knowledgebases = knowledgebase_results.all()
    logger.info(f"Listing knowledgebases: get {len(knowledgebases)} in total.")

    return success_response(data=knowledgebases, message="查询知识库成功。")


@knowledgebase_router.get(
    "/{kb_name}", response_model=ResponseModel[KnowledgebaseEntity]
)
async def read_knowledgebase(
    kb_name: str, session: AsyncSession = Depends(get_session)
):
    statement = select(KnowledgebaseEntity).where(KnowledgebaseEntity.name == kb_name)
    knowledgebase = (await session.exec(statement)).first()

    if not knowledgebase:
        return JSONResponse(
            content=error_response(code=404, message=f"查询知识库失败: 知识库'{kb_name}'不存在。"),
            status_code=404,
        )

    return success_response(data=knowledgebase, message="查询知识库成功。")


@knowledgebase_router.patch(
    "/{kb_name}", response_model=ResponseModel[KnowledgebaseEntity]
)
async def update_knowledgebase(
    kb_name: str,
    new_kb: KnowledgebaseCreate,
    session: AsyncSession = Depends(get_session),
):
    statement = select(KnowledgebaseEntity).where(KnowledgebaseEntity.name == kb_name)
    knowledgebase = (await session.exec(statement)).first()
    if not knowledgebase:
        return JSONResponse(
            content=error_response(code=404, message=f"更新知识库失败: 知识库'{kb_name}'不存在。"),
            status_code=404,
        )

    knowledgebase.name = new_kb.name or knowledgebase.name
    knowledgebase.description = new_kb.description or knowledgebase.description
    knowledgebase.doc_num = new_kb.doc_num or knowledgebase.doc_num
    knowledgebase.chunk_num = new_kb.chunk_num or knowledgebase.chunk_num
    if new_kb.chunk_config:
        knowledgebase.chunk_config = new_kb.chunk_config.model_dump()
    if new_kb.retrieval_config:
        knowledgebase.retrieval_config = new_kb.retrieval_config.model_dump()

    session.add(knowledgebase)
    await session.commit()
    await session.refresh(knowledgebase)

    asyncio.create_task(knowledgebase_provider.refresh())

    logger.info(f"Knowledgebase {kb_name} updated to {knowledgebase}.")

    return success_response(data=knowledgebase, message="更新知识库成功。")


@knowledgebase_router.delete("/{kb_name}")
async def delete_knowledgebase(
    kb_name: str,
    session: AsyncSession = Depends(get_session),
):
    statement = select(KnowledgebaseEntity).where(KnowledgebaseEntity.name == kb_name)
    knowledgebase = (await session.exec(statement)).first()

    if not knowledgebase:
        return JSONResponse(
            content=error_response(code=404, message=f"删除知识库失败: 知识库'{kb_name}'不存在。"),
            status_code=404,
        )

    await session.delete(knowledgebase)
    await session.commit()

    asyncio.create_task(mcp_provider.refresh())

    logger.info(f"Knowledgebase {knowledgebase} has been deleted.")

    return success_response(message=f"知识库'{kb_name}'删除成功。")
