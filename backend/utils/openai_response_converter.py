import json
import uuid
from llama_index.core.base.llms.types import (
    MessageRole,
    ChatResponseAsyncGen,
)
from loguru import logger
from openai.types.chat import ChatCompletionChunk
import time
from common.chat.models import ChatAgentRequest


class OpenAIChatCompletionChunkConverter:
    def __init__(self, chat_request: ChatAgentRequest):
        """Initialize the converter."""
        self.chat_model = chat_request.model

    def _make_json_chunk(self, content: any):
        """Helper function to format the content as a JSON chunk."""
        return json.dumps(content, ensure_ascii=False)

    async def aconvert_to_openai_chat_completion_chunk(
        self, async_response_gen: ChatResponseAsyncGen
    ):
        logger.info("Start generating chunks.")
        chat_id = uuid.uuid4().hex
        model = self.chat_model
        enable_citations = False
        citations = []
        citation_details = []
        chunk_id = 0
        async for response in async_response_gen:
            if response.message.role == MessageRole.ASSISTANT:
                tool_calls = response.message.additional_kwargs.get('tool_calls', [])
                if tool_calls and tool_calls[0].function.name in ["search-web", "search-knowledgebase"]:
                    enable_citations = True
                chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=int(time.time()),
                    model=model,
                    object="chat.completion.chunk",
                    choices=[
                        {
                            "index": chunk_id,
                            "delta": {
                                "content": response.delta,
                                "role": response.message.role,
                                "tool_calls": tool_calls,
                            },
                            "finish_reason": "stop" if response.message.additional_kwargs.get(
                                    "STOP_FLAG") else None,
                        },
                    ],
                )
                if response.message.additional_kwargs.get("STOP_FLAG"):
                    chunk.usage = response.additional_kwargs
                yield self._make_json_chunk(chunk.model_dump(mode="json"))
            elif response.message.role == MessageRole.TOOL:
                if enable_citations:
                    tool_call_results = json.loads(response.delta).get("result", [])
                    citations = [r["metadata"]["file_url"] for r in tool_call_results]
                    citation_details = [
                        {
                            "text": r["text"],
                            "name": r["metadata"]["file_name"],
                            "url": r["metadata"]["file_url"],
                            "score": r["score"],
                        }
                        for r in tool_call_results
                    ]
                chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=int(time.time()),
                    model=model,
                    object="chat.completion.chunk",
                    choices=[
                        {
                            "index": chunk_id,
                            "delta": {
                                "content": response.delta,
                                "role": response.message.role,
                                "tool_calls": response.message.additional_kwargs.get(
                                    "tool_calls", []
                                ),
                            },
                            "finish_reason": "stop" if response.message.additional_kwargs.get(
                                    "STOP_FLAG") else None,
                        },
                    ],
                )
                if response.message.additional_kwargs.get("STOP_FLAG"):
                    chunk.usage = response.additional_kwargs
                yield self._make_json_chunk(chunk.model_dump(mode="json"))
            else:
                raise ValueError(f"Unknown role: {response.message.role}")
            chunk_id += 1

        last_citation_chunk = ChatCompletionChunk(
            id=chat_id,
            created=int(time.time()),
            model=model,
            object="chat.completion.chunk",
            citations=citations,
            citation_details=citation_details,
            choices=[
                {
                    "index": chunk_id,
                    "delta": {
                        "content": "",
                        "role": MessageRole.ASSISTANT,
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                },
            ],
        )
        yield self._make_json_chunk(last_citation_chunk.model_dump(mode="json"))
        logger.info("Finished generating chunks.")
