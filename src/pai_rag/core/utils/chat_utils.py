from enum import Enum
from uuid import uuid4

import asyncio
import re
import time
import traceback
from typing import Any, AsyncGenerator, List

from openai import APIError
from pai_rag.app.api.models import ChatResponseWrapper, RagResponse
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ChatResponseAsyncGen,
    ChatResponse,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from llama_index.core.schema import ImageNode
from openai.types.chat.chat_completion import Choice
import openai.types.chat.chat_completion_chunk as chat_completion_chunk
import json

from loguru import logger


def chat_id_generator() -> str:
    return uuid4().hex


def parse_citations_from_source_nodes(
    response_wrapper: ChatResponseWrapper,
) -> List[str]:
    citations = []
    citation_details = []

    if not response_wrapper.source_nodes:
        return citations, citation_details

    for score_node in response_wrapper.source_nodes:
        if isinstance(score_node.node, ImageNode):
            url = score_node.node.image_url
            if url is not None:
                citations.append(url)
                citation_details.append(
                    {
                        "name": "Image",
                        "text": None,
                        "url": url,
                        "score": score_node.score,
                    }
                )
        else:
            url = score_node.node.metadata.get(
                "file_url"
            ) or score_node.node.metadata.get("file_path")
            citations.append(url)

            if score_node.node.metadata.get("invalid_flag") is not None:
                citation_details.append(
                    {
                        "name": "SQL Information",
                        "text": json.dumps(
                            {
                                "SQL": score_node.node.metadata.get(
                                    "query_code_instruction"
                                ),
                                "SQL_Exec_Result": score_node.node.text,
                                "Tables": score_node.node.metadata.get("query_tables"),
                                "Valid": score_node.node.metadata.get("invalid_flag"),
                            },
                            ensure_ascii=False,
                        ),
                        "url": url,
                        "score": score_node.score,
                    }
                )
            else:
                citation_details.append(
                    {
                        "name": score_node.node.metadata.get("file_name"),
                        "text": score_node.node.text,
                        "url": url,
                        "score": score_node.score,
                    }
                )

    return citations, citation_details


def get_token_usage(
    chat_response: ChatResponse, base_token_usage: CompletionUsage = None
) -> CompletionUsage:
    if base_token_usage is None:
        return CompletionUsage(
            completion_tokens=chat_response.additional_kwargs.pop(
                "completion_tokens", 0
            ),
            prompt_tokens=chat_response.additional_kwargs.pop("prompt_tokens", 0),
            total_tokens=chat_response.additional_kwargs.pop("total_tokens", 0),
        )
    else:
        return CompletionUsage(
            completion_tokens=chat_response.additional_kwargs.pop(
                "completion_tokens", 0
            )
            + base_token_usage.completion_tokens,
            prompt_tokens=chat_response.additional_kwargs.pop("prompt_tokens", 0)
            + base_token_usage.prompt_tokens,
            total_tokens=chat_response.additional_kwargs.pop("total_tokens", 0)
            + base_token_usage.total_tokens,
        )


def _make_json_chunk(data: Any):
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def make_completion_response(
    chat_id: str,
    model: str,
    response_wrapper: ChatResponseWrapper,
    base_token_usage: CompletionUsage,
    return_reference: bool = False,
):
    if isinstance(response_wrapper.response, str):
        return response_wrapper.response
    chat_response: ChatResponse = response_wrapper.response
    logger.info(f"Finished response: {chat_response.message.content}")
    token_usage = get_token_usage(chat_response, base_token_usage)

    citations, citation_details = [], []
    if return_reference:
        citations, citation_details = parse_citations_from_source_nodes(
            response_wrapper
        )

    return ChatCompletion(
        id=chat_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role=MessageRole.ASSISTANT.value,
                    content=chat_response.message.content,
                ),
                finish_reason="stop",
            )
        ],
        citation_details=citation_details,
        citations=citations,
        object="chat.completion",
        usage=token_usage,
        **chat_response.additional_kwargs,
    )


async def make_completion_chunk_response(
    chat_id: str,
    model: str,
    response_wrapper: ChatResponseWrapper,
    base_token_usage: CompletionUsage,
    start_time: float = 0,
    return_reference: bool = False,
) -> AsyncGenerator[str, None]:
    chunk_id = 0
    full_content = ""
    created_ts = int(time.time())

    if isinstance(response_wrapper.response, str):
        yield response_wrapper.response
        return
    else:
        try:
            is_first_chunk = True
            chunk_token_usage = None

            chat_response_gen: ChatResponseAsyncGen = response_wrapper.response
            citations, citation_details = [], []
            if return_reference:
                citations, citation_details = parse_citations_from_source_nodes(
                    response_wrapper
                )
            async for chat_response in chat_response_gen:
                chunk_token_usage = get_token_usage(chat_response, base_token_usage)
                if not chat_response.delta and not chat_response.additional_kwargs:
                    continue

                if is_first_chunk:
                    if len(citations) > 0:
                        chat_response.additional_kwargs["citations"] = citations
                        chat_response.additional_kwargs[
                            "citation_details"
                        ] = citation_details
                    if chat_response.delta:
                        logger.info(
                            f"[{chat_id}] Start get first token {time.time() - start_time}"
                        )

                    is_first_chunk = False

                full_content += chat_response.delta
                chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=created_ts,
                    model=model,
                    choices=[
                        chat_completion_chunk.Choice(
                            index=chunk_id,
                            delta=chat_completion_chunk.ChoiceDelta(
                                role=MessageRole.ASSISTANT.value,
                                content=chat_response.delta,
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=chunk_token_usage,
                    object="chat.completion.chunk",
                    **chat_response.additional_kwargs,
                )
                chunk_id += 1
                yield _make_json_chunk(data=chunk.model_dump(mode="json"))

            last_chunk = ChatCompletionChunk(
                id=chat_id,
                created=created_ts,
                model=model,
                citations=citations,
                citation_details=citation_details,
                choices=[
                    chat_completion_chunk.Choice(
                        index=chunk_id,
                        delta=chat_completion_chunk.ChoiceDelta(
                            role=MessageRole.ASSISTANT.value,
                            content="",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=chunk_token_usage,
                object="chat.completion.chunk",
            )
            yield _make_json_chunk(data=last_chunk.model_dump(mode="json"))
            logger.info(f"Finished streaming: {full_content}")
        except APIError as exception:
            logger.error(f"Streaming failed: {traceback.format_exc()}")
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=created_ts,
                model=model,
                choices=[
                    chat_completion_chunk.Choice(
                        index=chunk_id,
                        delta=chat_completion_chunk.ChoiceDelta(
                            role=MessageRole.ASSISTANT.value,
                            content=exception.message,
                        ),
                        finish_reason="stop",
                    )
                ],
                object="chat.completion.chunk",
            )
            yield _make_json_chunk(data=chunk.model_dump(mode="json"))
        except asyncio.CancelledError:
            logger.warning(f"Streaming cancelled: {chat_id} {full_content}")
        except Exception as exception:
            logger.info(f"Streaming failed: {exception}")
            raise exception


def response_from_text(text: str):
    return ChatResponseWrapper(
        response=ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT.value,
                content=text,
            ),
            additional_kwargs={},
        )
    )


def response_gen_from_text(text: str):
    async def text_gen():
        yield ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT.value,
                content=text,
            ),
            additional_kwargs={},
            delta=text,
        )

    return ChatResponseWrapper(
        response=text_gen(),
        additional_kwargs={},
    )


class SseVersion(int, Enum):
    V0 = 0  # Backward compatibility
    V1 = 1  # New V1 version


def _event_chunk_wrapper(chunk_content, sse_version: SseVersion = SseVersion.V1):
    if sse_version == sse_version.V1:
        return f"data: {chunk_content}\n\n"
    else:
        return f"{chunk_content}\n"


def make_legacy_response(
    response_wrapper: ChatResponseWrapper,
    history_messages=[],
    docs=[],
    chat_store=None,
    session_id=None,
) -> RagResponse:
    message_content = response_wrapper.response.message.content
    if chat_store:
        history_content = re.sub(
            r"<think>.*?</think>\n*", "", message_content, flags=re.DOTALL
        )
        history_content = history_content.replace("<think>", "").replace("</think>", "")
        history_messages.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=history_content,
            )
        )
        chat_store.set_messages(session_id, history_messages)

    return RagResponse(
        answer=message_content,
        session_id=session_id,
        docs=docs,
    )


async def make_legacy_sse_chunk_async(
    response_wrapper: ChatResponseWrapper,
    history_messages=[],
    docs=[],
    chat_store=None,
    session_id=None,
    sse_version: SseVersion = SseVersion.V0,
):
    message_content = ""
    async for chat_response in response_wrapper.response:
        if chat_response.delta:
            chunk = {"delta": chat_response.delta, "is_finished": False}
            message_content += chat_response.delta
            yield _event_chunk_wrapper(
                json.dumps(chunk, ensure_ascii=False), sse_version
            )

    if chat_store:
        message_content = re.sub(
            r"<think>.*?</think>\n*", "", message_content, flags=re.DOTALL
        )
        message_content = message_content.replace("<think>", "").replace("</think>", "")
        history_messages.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=message_content,
            )
        )
        chat_store.set_messages(session_id, history_messages)

    if docs:
        # 返回
        last_chunk = {"delta": "", "is_finished": True, "docs": docs}
    else:
        last_chunk = {"delta": "", "is_finished": True}

    last_chunk_data = json.dumps(
        last_chunk, default=lambda x: x.dict(), ensure_ascii=False
    )
    yield _event_chunk_wrapper(last_chunk_data, sse_version)
