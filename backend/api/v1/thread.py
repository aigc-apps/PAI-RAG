from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import get_session
from loguru import logger
from db.models.thread import ThreadEntity, ThreadCreate, ThreadRead
from db.models.message import MessageEntity, MessageCreate, MessageRead
from sqlalchemy.exc import IntegrityError
from typing import List
from sqlmodel import select
from db.models.knowledgebase.file import KbFileEntity
from llama_index.core.llms import LLM
from config.providers.llm_provider import llm_provider
from db.models.llm import LlmModelEntity
from utils.message_utils import convert_to_chat_messages

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
    sql_results = await session.exec(select(ThreadEntity).order_by(ThreadEntity.created_at.desc()).offset(offset).limit(limit))
    thread_entities = sql_results.all()
    thread_models = [
        ThreadRead.model_validate(
            thread,
        )
        for thread in thread_entities
    ]
    return thread_models

async def delete_related_attachments_in_messages(session: AsyncSession, thread_id: str):
    logger.info("[thread] Start deleting related attachments in messages.")
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
    message_ids = [message.id for message in message_models]
    file_sql_results = await session.exec(
        select(KbFileEntity)
        .where(KbFileEntity.message_id.in_(message_ids))
    )
    attachment_file_entities = file_sql_results.all()
    for attachment_file_entity in attachment_file_entities:
        await session.delete(attachment_file_entity)
        await session.commit()
    logger.info("[thread] Deleted related attachments in messages successfully.")

@thread_router.delete("/{thread_id}")
async def delete_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
):
    try:
        await delete_related_attachments_in_messages(session, thread_id)
    except Exception as e:
        logger.error(f"[ThreadProvider] Failed to delete related attachments in messages: {e}")
    thread = await session.get(ThreadEntity, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Conversation {thread_id} not found.")
    await session.delete(thread)
    await session.commit()

    logger.info(f"Conversation {thread_id} deleted.")
    return {"message": f"Conversation {thread_id} deleted."}

@thread_router.patch("/{thread_id}")
async def update_thread_title(
    thread_id: str,
    messages: List[MessageCreate],
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Updating conversation {thread_id} title based on messages {messages}.")
    thread = await session.get(ThreadEntity, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Conversation {thread_id} not found.")
    try:
        llm_sql_results = await session.exec(select(LlmModelEntity))
        llm_entities = llm_sql_results.all()
        llm: LLM = llm_provider.get_llm_model(model_id=llm_entities[0].model_id)
        input_messages = [{"role": "system", "content": "你是一个智能助手，负责为对话生成标题，输出5-10个字即可。"} ] + [msg.model_dump() for msg in messages]
        chat_response = await llm.achat(
            messages=convert_to_chat_messages(input_messages),
        )
        thread.title = chat_response.message.content
    except Exception as e:
        logger.error(f"Failed to update conversation {thread_id} title: {e}")
        thread.title = f"{messages[0].content[0]['text'][:10]}..." if messages else "未命名会话"

    session.add(thread)
    await session.commit()

    logger.info(f"Conversation {thread_id} updated.")
    return {"message": f"Conversation {thread_id} updated."}

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
        attachment_file_entity = await session.get(KbFileEntity, attachment.get("id"))
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
