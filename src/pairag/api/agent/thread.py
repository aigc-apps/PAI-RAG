import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.db_context import get_session
from loguru import logger
from pairag.db.models.thread import ThreadEntity, ThreadCreate, ThreadRead
from pairag.db.models.message import MessageEntity, MessageCreate, MessageRead
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.thread_provider import thread_provider
from typing import List
from sqlmodel import select
from pairag.db.models.attachment.file import AttachmentFileEntity

thread_router = APIRouter()


@thread_router.post("", response_model=ThreadRead)
async def create_thread(
    thread: ThreadCreate, session: AsyncSession = Depends(get_session)
):
    try:
        thread = ThreadEntity.model_validate(thread)
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        asyncio.create_task(thread_provider.refresh())
        return thread

    except IntegrityError as e:
        logger.exception(f"Failed to add conversation: {e}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            raise HTTPException(
                status_code=400, detail=f"Conversation {thread} already exists."
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Failed to add conversation: {str(e)}"
            )
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to add conversation: {str(e)}")


@thread_router.get("", response_model=List[ThreadRead])
async def get_threads(
    session: AsyncSession = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    sql_results = await session.exec(select(ThreadEntity).offset(offset).limit(limit))
    thread_entities = sql_results.all()
    thread_models = [
        ThreadRead.model_validate(
            thread,
        )
        for thread in thread_entities
    ]
    return thread_models


@thread_router.delete("/{thread_id}")
async def delete_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
):
    thread = await session.get(ThreadEntity, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Conversation {thread_id} not found.")
    await session.delete(thread)
    await session.commit()
    asyncio.create_task(thread_provider.refresh())

    logger.info(f"Conversation {thread_id} deleted.")
    return {"message": f"Conversation {thread_id} deleted."}


@thread_router.post("/{thread_id}/messages")
async def create_thread_message(
    message: MessageCreate,
    session: AsyncSession = Depends(get_session),
):
    thread_id = message.thread_id
    thread = await session.get(ThreadEntity, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Conversation {thread_id} not found.")

    message_entity = MessageEntity.model_validate(message)

    session.add(message_entity)
    await session.commit()
    await session.refresh(message_entity)

    for attachment in message.attachments:
        file_res = await session.exec(
            select(AttachmentFileEntity).where(
                AttachmentFileEntity.frontend_file_id == attachment.get("id")
            )
        )
        attachment_file_entity = file_res.first()
        attachment_file_entity.message_id = message_entity.id
        session.add(attachment_file_entity)
        await session.commit()
    await session.flush()

    return message_entity


@thread_router.get("/{thread_id}/messages", response_model=List[MessageRead])
async def get_thread_messages(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
):
    sql_results = await session.exec(
        select(MessageEntity)
        .where(MessageEntity.thread_id == thread_id)
    )
    message_entities = sql_results.all()
    message_models = [
        MessageRead.model_validate(
            message,
        )
        for message in message_entities
    ]
    return message_models
