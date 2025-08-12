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
import re
from typing import Tuple


THINKING_REGEX = re.compile(r"^<think>\n(.*?)\n</think>\n")
THINKING_START_REGEX = re.compile(r"^<think>\n")

class OpenAIChatCompletionChunkConverter:
    def __init__(self, chat_request: ChatAgentRequest):
        """Initialize the converter."""
        self.chat_model = chat_request.model

    def _make_json_chunk(self, content: any):
        """Helper function to format the content as a JSON chunk."""
        return json.dumps(content, ensure_ascii=False)

    def separate_thinking(self, response: str) -> Tuple[str, str]:
        """Separate the thinking from the response."""
        # 提取所有完整的 <think>...</think> 内容
        thinking_parts = []
        clean_response = response

        # 循环提取所有完整的 thinking 标签
        while True:
            match = re.search(r"<think>(.*?)</think>", clean_response, re.DOTALL)
            if match:
                thinking_parts.append(match.group(1))
                # 移除这个完整的标签
                clean_response = clean_response[:match.start()] + clean_response[match.end():]
            else:
                break

        thinking_content = "".join(thinking_parts)
        return thinking_content, clean_response

    async def aconvert_to_openai_chat_completion_chunk(
        self, async_response_gen: ChatResponseAsyncGen
    ):
        logger.info("Start generating chunks.")
        chat_id = uuid.uuid4().hex
        model = self.chat_model
        chunk_id = 0
        previous_assistant_message = None
        citations = []
        citation_details = []
        full_content = ""  # 完整内容用于提取 thinking
        last_clean_content = ""  # 上一次的清理后内容
        async for response in async_response_gen:
            if response.message.role == MessageRole.ASSISTANT:
                previous_assistant_message = response.message
                 # 累积完整内容
                full_content += (response.delta or "")

                # 从完整内容中提取 thinking 和清理后的内容
                accumulated_thinking, clean_content = self.separate_thinking(full_content)

                # 计算这次要发送的增量内容（delta）
                delta_content = clean_content[len(last_clean_content):]
                last_clean_content = clean_content

                print(f"Accumulated thinking: {accumulated_thinking}")
                print(f"Delta content: '{delta_content}'")

                chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=int(time.time()),
                    model=model,
                    object="chat.completion.chunk",
                    choices=[
                        {
                            "index": chunk_id,
                            "delta": {
                                "content": delta_content,
                                "role": response.message.role,
                                "tool_calls": response.message.additional_kwargs.get('tool_calls', []),
                                "reasoning_content": accumulated_thinking or response.message.additional_kwargs.get('reasoning_content', '')
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
                # If the response is from a tool, we check if it contains citations
                if previous_assistant_message:
                    previous_tool_calls = previous_assistant_message.additional_kwargs.get('tool_calls', [])
                    if previous_tool_calls and previous_tool_calls[0].function.name in ["search-web", "search-knowledgebase"]:
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

        if len(citations) > 0:
            logger.info("Generating last citation chunk.")
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
