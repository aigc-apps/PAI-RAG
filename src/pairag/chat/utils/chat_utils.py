import asyncio
import time
import traceback
from typing import Any, AsyncGenerator, List
import uuid

from pairag.chat.models import ChatResponseWrapper
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

from pairag.integrations.query_transform.intent_models import ChatIntentType
from asgi_correlation_id import correlation_id

DEFAULT_ERROR_RESPONSE = "抱歉，系统出错，暂时无法处理这个请求。"


def chat_id_generator() -> str:
    return correlation_id.get() or uuid.uuid4().hex


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
    return json.dumps(data, ensure_ascii=False)


def make_completion_response(
    chat_id: str,
    model: str,
    response_wrapper: ChatResponseWrapper,
    return_reference: bool = False,
):
    chat_response: ChatResponse = response_wrapper.response
    logger.info(f"Finished response: {chat_response.message.content}")
    if response_wrapper.intent_result:
        token_usage = get_token_usage(
            chat_response, response_wrapper.intent_result.token_usage
        )
    else:
        token_usage = get_token_usage(chat_response)

    if (
        response_wrapper.intent_result is not None
        and response_wrapper.intent_result.intent != ChatIntentType.CHAT_NEWS
    ):
        chat_response.additional_kwargs[
            "intent"
        ] = response_wrapper.intent_result.intent

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
    start_time: float = 0,
    return_reference: bool = False,
) -> AsyncGenerator[str, None]:
    chunk_id = 0
    full_content = ""
    created_ts = int(time.time())

    try:
        is_first_chunk = True
        chunk_token_usage = None

        chat_response_gen: ChatResponseAsyncGen = response_wrapper.response
        citations, citation_details = [], []
        if return_reference:
            citations, citation_details = parse_citations_from_source_nodes(
                response_wrapper
            )

        if (
            response_wrapper.intent_result is not None
            and response_wrapper.intent_result.intent != ChatIntentType.CHAT_NEWS
        ):
            intent_kwargs = {
                "intent": response_wrapper.intent_result.intent,
            }

            intent_chunk = ChatCompletionChunk(
                id=chat_id,
                created=created_ts,
                model=model,
                choices=[
                    chat_completion_chunk.Choice(
                        index=chunk_id,
                        delta=chat_completion_chunk.ChoiceDelta(
                            role=MessageRole.ASSISTANT.value,
                            content="",
                        ),
                        finish_reason=None,
                    )
                ],
                usage=chunk_token_usage,
                object="chat.completion.chunk",
                **intent_kwargs,
            )
            yield _make_json_chunk(data=intent_chunk.model_dump(mode="json"))
            chunk_id += 1

        async for chat_response in chat_response_gen:
            if response_wrapper.intent_result:
                chunk_token_usage = get_token_usage(
                    chat_response, response_wrapper.intent_result.token_usage
                )
            else:
                chunk_token_usage = get_token_usage(chat_response)
            if not chat_response.delta and not chat_response.additional_kwargs:
                continue

            if is_first_chunk:
                if len(citations) > 0:
                    chat_response.additional_kwargs["citations"] = citations
                    chat_response.additional_kwargs[
                        "citation_details"
                    ] = citation_details
                if chat_response.delta:
                    logger.info(f"Start get first token {time.time() - start_time}")

                is_first_chunk = False

            full_content += chat_response.delta
            if "prompt_tokens" not in chat_response.additional_kwargs:
                chat_response.additional_kwargs[
                    "prompt_tokens"
                ] = chunk_token_usage.prompt_tokens

            if "completion_tokens" not in chat_response.additional_kwargs:
                chat_response.additional_kwargs[
                    "completion_tokens"
                ] = chunk_token_usage.completion_tokens

            if "total_tokens" not in chat_response.additional_kwargs:
                chat_response.additional_kwargs[
                    "total_tokens"
                ] = chunk_token_usage.total_tokens

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
    except asyncio.CancelledError:
        logger.warning(f"Streaming cancelled: {full_content}")
        raise
    except (BrokenPipeError, ConnectionError) as e:
        # Client already went away; do not attempt another yield.
        logger.warning(f"Streaming aborted, client disconnected: {e}")
        raise
    except Exception:
        logger.error(f"Streaming failed: {traceback.format_exc()}")
        try:
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=created_ts,
                model=model,
                choices=[
                    chat_completion_chunk.Choice(
                        index=chunk_id,
                        delta=chat_completion_chunk.ChoiceDelta(
                            role=MessageRole.ASSISTANT.value,
                            content=DEFAULT_ERROR_RESPONSE,
                        ),
                        finish_reason="stop",
                    )
                ],
                object="chat.completion.chunk",
            )
            yield _make_json_chunk(data=chunk.model_dump(mode="json"))
        except (BrokenPipeError, ConnectionError, asyncio.CancelledError) as e:
            # Writing the fallback chunk failed because the peer is gone.
            # Swallow the secondary error so the original traceback above is
            # what surfaces in logs; the outer generator machinery will
            # propagate as needed.
            logger.warning(
                f"Failed to deliver error chunk, client disconnected: {e}"
            )
        except Exception:
            logger.error(
                f"Failed to deliver error chunk: {traceback.format_exc()}"
            )

        raise


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
