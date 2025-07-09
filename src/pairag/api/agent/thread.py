import asyncio
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.db_context import get_session
from loguru import logger
from pairag.api.response_model import ResponseModel, success_response
from pairag.db.models.thread import ThreadEntity, ThreadCreate
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.thread_provider import thread_provider

thread_router = APIRouter()


@thread_router.post("", response_model=ResponseModel[ThreadEntity])
async def create_thread(
    thread: ThreadCreate, session: AsyncSession = Depends(get_session)
):
    try:
        thread = ThreadEntity.model_validate(thread)
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        asyncio.create_task(thread_provider.refresh())
        return success_response(data=thread, message="Thread创建成功。")

    except IntegrityError as e:
        logger.exception(f"创建Thread失败。\nIntegrityError:{e}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            raise HTTPException(
                status_code=400, detail=f"Thread {thread} already exists."
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Failed to add thread: {str(e)}"
            )
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to add thread: {str(e)}")
