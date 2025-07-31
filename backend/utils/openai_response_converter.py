import json
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
        return f"data:{json.dumps(content, ensure_ascii=False)}\n"

    async def aconvert_to_openai_chat_completion_chunk(
        self, async_response_gen: ChatResponseAsyncGen
    ):
        logger.info("Start generating chunks.")
        chat_id = "chat_test_id_0"
        model = self.chat_model
        async for response in async_response_gen:
            chunk_id = 0
            if response.message.role == MessageRole.ASSISTANT:
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
                        },
                    ],
                )
                yield self._make_json_chunk(chunk.model_dump(mode="json"))
            elif response.message.role == MessageRole.TOOL:
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
                        },
                    ],
                )
                yield self._make_json_chunk(chunk.model_dump(mode="json"))
            else:
                raise ValueError(f"Unknown role: {response.message.role}")
            chunk_id += 1

        logger.info("Finished generating chunks.")
