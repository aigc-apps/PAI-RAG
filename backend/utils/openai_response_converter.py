from asyncio import CancelledError
import json
import uuid
from llama_index.core.base.llms.types import (
    MessageRole,
    ChatResponseAsyncGen,
)
from loguru import logger
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice
import time
from common.chat.models import ChatAgentRequest


def extract_citation_url(record: dict) -> str:
    return record["metadata"].get("file_source") or record["metadata"].get("file_url")


def tool_has_citation(tool_name: str) -> bool:
    return (
        tool_name == "search-web" or
        tool_name.startswith("search-knowledgebase")
    )


def get_citation_source(tool_name: str) -> str:
    return "web" if tool_name == "search-web" else "knowledgebase"


class OpenAIChatCompletionConverter:
    def __init__(self, chat_request: ChatAgentRequest):
        """Initialize the converter."""
        self.chat_model = chat_request.model

    def _make_json_chunk(self, content: any):
        """Helper function to format the content as a JSON chunk."""
        return json.dumps(content, ensure_ascii=False)

    async def aconvert(
        self,
        async_response_gen: ChatResponseAsyncGen,
    ):
        logger.info("Start generating response.")
        chat_id = uuid.uuid4().hex
        usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        model = self.chat_model
        safety_violation = False
        previous_assistant_message = None
        citations = []
        citation_details = []
        full_content = ""
        try:
            async for response in async_response_gen:
                if response.message.role == MessageRole.ASSISTANT:
                    safety_violation = response.additional_kwargs.get("safety_violation", False)
                    usage.prompt_tokens += response.additional_kwargs.get("prompt_tokens", 0)
                    usage.completion_tokens += response.additional_kwargs.get("completion_tokens", 0)
                    usage.total_tokens += response.additional_kwargs.get("total_tokens", 0)

                    previous_assistant_message = response.message
                    full_content += response.delta or ""
                elif response.message.role == MessageRole.TOOL:
                    # If the response is from a tool, we check if it contains citations
                    if previous_assistant_message:
                        previous_tool_calls = previous_assistant_message.additional_kwargs.get('tool_calls', [])
                        if previous_tool_calls and tool_has_citation(previous_tool_calls[0].function.name):
                            tool_call_results = json.loads(response.delta).get("result", [])
                            citations = [extract_citation_url(r) for r in tool_call_results]
                            citation_details = [
                                {
                                    "source": get_citation_source(previous_tool_calls[0].function.name),
                                    "text": r["text"],
                                    "name": r["metadata"]["file_name"],
                                    "url": extract_citation_url(r),
                                    "score": r["score"],
                                }
                                for r in tool_call_results
                            ]
        except CancelledError:
            logger.warning("Request cancelled by client.")

        logger.info(f"Finished generating response: {full_content}.")
        return ChatCompletion(
            id=chat_id,
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role=MessageRole.ASSISTANT.value,
                        content=full_content,
                    ),
                    finish_reason="stop",
                )
            ],
            citation_details=citation_details,
            citations=citations,
            object="chat.completion",
            usage=usage,
            safety_violation=safety_violation,
        )


    async def astream_convert(
        self,
        async_response_gen: ChatResponseAsyncGen,
    ):
        logger.info("Start streaming response.")
        chat_id = uuid.uuid4().hex
        usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        model = self.chat_model
        chunk_id = 0
        previous_assistant_message = None
        citations = []
        citation_details = []
        full_content = ""
        try:
            async for response in async_response_gen:
                if response.message.role == MessageRole.ASSISTANT:
                    usage.prompt_tokens += response.additional_kwargs.get("prompt_tokens", 0)
                    usage.completion_tokens += response.additional_kwargs.get("completion_tokens", 0)
                    usage.total_tokens += response.additional_kwargs.get("total_tokens", 0)

                    previous_assistant_message = response.message

                    is_last_chunk = True if response.additional_kwargs.get("STOP_FLAG") else False

                    actions = [
                        action.model_dump(mode="json") for action in
                        response.message.additional_kwargs.get('tool_calls', [])
                    ]
                    full_content += response.delta or ""
                    if is_last_chunk:
                        chunk = ChatCompletionChunk(
                            id=chat_id,
                            created=int(time.time()),
                            model=model,
                            object="chat.completion.chunk",
                            actions=actions,
                            safety_violation=response.additional_kwargs.get("safety_violation", False),
                            choices=[
                                {
                                    "index": chunk_id,
                                    "delta": {
                                        "content": response.delta,
                                        "role": response.message.role,
                                        "tool_calls": [],
                                        "reasoning_content": (
                                            response.raw.choices[0].delta.reasoning_content
                                            if response.raw and response.raw.choices and hasattr(response.raw.choices[0].delta, 'reasoning_content')
                                            else ""
                                        ),
                                        "reasoning_completed": response.additional_kwargs.get("reasoning_completed", False),
                                    },
                                    "finish_reason": "stop"
                                },
                            ],
                            citations=citations,
                            citation_details=citation_details,
                            usage=usage,
                        )
                    else:
                        chunk = ChatCompletionChunk(
                            id=chat_id,
                            created=int(time.time()),
                            model=model,
                            object="chat.completion.chunk",
                            actions=actions,
                            safety_violation=response.additional_kwargs.get("safety_violation", False),
                            choices=[
                                {
                                    "index": chunk_id,
                                    "delta": {
                                        "content": response.delta,
                                        "role": response.message.role,
                                        "tool_calls": [],
                                        "reasoning_content": (
                                            response.raw.choices[0].delta.reasoning_content
                                            if response.raw and response.raw.choices and hasattr(response.raw.choices[0].delta, 'reasoning_content')
                                            else ""
                                        ),
                                        "reasoning_completed": response.additional_kwargs.get("reasoning_completed", False),
                                    },
                                    "finish_reason": None
                                },
                            ]
                        )
                    yield self._make_json_chunk(chunk.model_dump(mode="json"))
                elif response.message.role == MessageRole.TOOL:
                    # If the response is from a tool, we check if it contains citations
                    if previous_assistant_message:
                        previous_tool_calls = previous_assistant_message.additional_kwargs.get('tool_calls', [])
                        if previous_tool_calls and tool_has_citation(previous_tool_calls[0].function.name):
                            tool_call_results = json.loads(response.delta).get("result", [])
                            citations = [extract_citation_url(r) for r in tool_call_results]
                            citation_details = [
                                {
                                    "source": get_citation_source(previous_tool_calls[0].function.name),
                                    "text": r["text"],
                                    "name": r["metadata"]["file_name"],
                                    "url": extract_citation_url(r),
                                    "score": r["score"],
                                }
                                for r in tool_call_results
                            ]
                    chunk = ChatCompletionChunk(
                        id=chat_id,
                        created=int(time.time()),
                        model=model,
                        object="chat.completion.chunk",
                        observation=response.delta,
                        choices=[
                            {
                                "index": chunk_id,
                                "delta": {
                                    "content": "",
                                    "role": MessageRole.ASSISTANT,
                                    "tool_calls": response.message.additional_kwargs.get(
                                        "tool_calls", []
                                    ),
                                    "reasoning_content":(
                                        response.raw.choices[0].delta.reasoning_content
                                        if response.raw and response.raw.choices and hasattr(response.raw.choices[0].delta, 'reasoning_content')
                                        else ""
                                    ),
                                    "reasoning_completed": response.additional_kwargs.get("reasoning_completed", False),
                                },
                                "finish_reason": None,
                            },
                        ],
                    )
                    yield self._make_json_chunk(chunk.model_dump(mode="json"))
                else:
                    raise ValueError(f"Unknown role: {response.message.role}")
                chunk_id += 1
        except CancelledError:
            logger.warning("Request cancelled by client.")

        logger.info(f"Finished generating chunks: {full_content}")
        return
