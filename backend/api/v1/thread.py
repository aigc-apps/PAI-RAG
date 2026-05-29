import re
import json
from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import get_db_session
from loguru import logger
from db.models.thread import ThreadCreate, ThreadRead
from db.models.message import MessageCreate, MessageRead
from typing import List
from common.chat.response_model import success_response, ResponseModel
from api.api_exception import ApiException, handle_api_exceptions
from common.chat.prompts import DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE
from utils.message_utils import get_content_from_messages
from llama_index.core.base.llms.types import MessageRole
from service.factory.model_factory import create_llm
from service.thread.thread_service import ThreadService
from service.thread.message_service import MessageService
from service.model.llm_service import LlmService
from service.injection import (
    get_thread_service,
    get_message_service,
    get_llm_service,
    get_tenant_id,
)

thread_router = APIRouter()


@thread_router.post("", response_model=ResponseModel[ThreadRead])
@handle_api_exceptions(action="create conversation", default_code=400)
async def create_thread(
    thread: ThreadCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    thread_service: ThreadService = Depends(get_thread_service),
):
    thread_entity = await thread_service.create_thread(thread, tenant_id=tenant_id)
    await session.commit()
    await session.refresh(thread_entity)
    return success_response(
        data=thread_entity, message="Conversation created successfully."
    )


@thread_router.get("", response_model=ResponseModel[List[ThreadRead]])
@handle_api_exceptions(action="list conversations", default_code=400)
async def get_threads(
    session: AsyncSession = Depends(get_db_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
    tenant_id: str = Depends(get_tenant_id),
    thread_service: ThreadService = Depends(get_thread_service),
):
    thread_entities = await thread_service.list_threads(
        tenant_id=tenant_id,
        offset=offset,
        limit=limit,
    )
    thread_models = [
        ThreadRead.model_validate(thread) for thread in thread_entities
    ]
    return success_response(
        data=thread_models, message="Get conversations successfully."
    )

@thread_router.delete("/{thread_id}")
@handle_api_exceptions(action="delete conversation", value_error_code=404, default_code=400)
async def delete_thread(
    thread_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    thread_service: ThreadService = Depends(get_thread_service),
    message_service: MessageService = Depends(get_message_service),
):
    # Release file refs FIRST. Any error here must abort the whole request
    # — otherwise the thread is gone but ref_counts stay elevated with no
    # way to reach them, and the files would be pinned forever.
    await message_service.release_attachment_refs(thread_id, tenant_id=tenant_id)

    # Delete the thread
    await thread_service.delete_thread(thread_id, tenant_id=tenant_id)
    await session.commit()

    return success_response(
        code=200, message=f"Conversation {thread_id} deleted."
    )

@thread_router.post("/{thread_id}/title")
@handle_api_exceptions(action="update conversation title", default_code=400)
async def update_thread_title(
    thread_id: str,
    messages: List[MessageCreate],
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    thread_service: ThreadService = Depends(get_thread_service),
    llm_service: LlmService = Depends(get_llm_service),
):
    logger.info(
        f"Updating conversation {thread_id} title based on messages {messages}."
    )
    thread = await thread_service.get_thread(thread_id, tenant_id=tenant_id)
    if not thread:
        raise ApiException(
            code=404, message=f"Conversation {thread_id} not found."
        )

    title = "未命名会话"
    try:
        # Get first LLM model from database
        llm_entities = await llm_service.get_all_llms(tenant_id=tenant_id)
        if not llm_entities:
            raise ValueError("No LLM models found in database.")

        llm_model = llm_entities[0]
        llm = create_llm(llm_model)

        # Generate title using LLM
        generate_title_prompt = DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE.format(
            chat_history="\n".join(
                [
                    f"{msg.role}: {get_content_from_messages(msg.content)}"
                    for msg in messages
                ]
            )
        )
        chat_response_gen = await llm.astream(
            messages=[
                {
                    "role": MessageRole.USER,
                    "content": generate_title_prompt,
                }
            ]
        )
        response_text = ""
        async for chunk in chat_response_gen:
            response_text += chunk.delta

        extracted_content = re.sub(
            r"<think>.*?</think>",
            "",
            response_text,
            flags=re.DOTALL,
        )
        logger.info(f"Generated title: {extracted_content}")
        title = json.loads(extracted_content).get("title", "未命名会话")
    except Exception as e:
        logger.error(f"Failed to update conversation {thread_id} title: {e}")
        # Fallback to simple title
        title = (
            f"{get_content_from_messages(messages[0].content)[:10]}..."
            if messages
            else "未命名会话"
        )

    # Update thread title
    await thread_service.update_thread_title(thread_id, title, tenant_id=tenant_id)
    await session.commit()

    logger.info(f"Conversation {thread_id} updated title to {title}.")
    return success_response(
        data={"title": title},
        message=f"Conversation {thread_id} title updated successfully.",
    )

@thread_router.post("/{thread_id}/messages", response_model=ResponseModel[MessageRead])
@handle_api_exceptions(action="create message", default_code=400)
async def create_thread_message(
    message: MessageCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    thread_service: ThreadService = Depends(get_thread_service),
    message_service: MessageService = Depends(get_message_service),
):
    thread_id = message.thread_id
    # Verify thread exists
    thread = await thread_service.get_thread(thread_id, tenant_id=tenant_id)
    if not thread:
        raise ApiException(
            code=404, message=f"Conversation {thread_id} not found."
        )

    # Create or update message
    message_entity = await message_service.create_message(message, tenant_id=tenant_id)
    await session.refresh(message_entity)

    return success_response(
        data=message_entity, message="Message created successfully."
    )


@thread_router.get("/{thread_id}/messages", response_model=ResponseModel[List[MessageRead]])
@handle_api_exceptions(action="retrieve messages", default_code=400)
async def get_thread_messages(
    thread_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    message_service: MessageService = Depends(get_message_service),
):
    message_entities = await message_service.list_messages(
        thread_id=thread_id,
        tenant_id=tenant_id,
        limit=30,
    )
    message_models = [
        MessageRead.model_validate(message) for message in message_entities
    ]
    return success_response(
        data=message_models, message="Messages retrieved successfully."
    )
