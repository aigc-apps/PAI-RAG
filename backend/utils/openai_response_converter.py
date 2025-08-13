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
from typing import Optional, List
from enum import Enum
from dataclasses import dataclass


THINKING_REGEX = re.compile(r"^<think>\n(.*?)\n</think>\n")
THINKING_START_REGEX = re.compile(r"^<think>\n")


class ChunkType(Enum):
    TEXT_DELTA = "text_delta"
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_COMPLETE = "thinking_complete"
@dataclass
class ExtractionResult:
    content: str
    is_tag_content: bool
    complete: bool = False
    tag_content_extracted: Optional[str] = None

@dataclass
class TagConfig:
    opening_tag: str
    closing_tag: str
    separator: str = "\n"


class TagExtractor:
    def __init__(self, config: TagConfig):
        self.config = config
        self.buffer = ""
        self.in_tag = False
        self.last_extracted_thinking = ""  # 记录上次提取的思考内容


    def process_text(self, text: str) -> List[ExtractionResult]:
        results = []
        self.buffer += text
        while self.buffer:
            if not self.in_tag:
                # 查找开始标签
                start_pos = self.buffer.find(self.config.opening_tag)

                # FIX: Check for stray closing tags and skip them
                end_pos = self.buffer.find(self.config.closing_tag)
                while end_pos != -1 and (start_pos == -1 or end_pos < start_pos):
                    # Skip over the closing tag
                    self.buffer = self.buffer[end_pos + len(self.config.closing_tag):]
                    end_pos = self.buffer.find(self.config.closing_tag)

                # Now look for the opening tag again with cleaned buffer
                start_pos = self.buffer.find(self.config.opening_tag)

                if start_pos != -1:
                    # 有开始标签，先输出开始标签之前的内容
                    if start_pos > 0:
                        results.append(ExtractionResult(
                            content=self.buffer[:start_pos],
                            is_tag_content=False,
                            complete=False
                        ))
                    # 进入标签内状态
                    self.in_tag = True
                    self.buffer = self.buffer[start_pos + len(self.config.opening_tag):]
                else:
                    # 没有开始标签，全部作为普通内容输出
                    results.append(ExtractionResult(
                        content=self.buffer,
                        is_tag_content=False,
                        complete=False
                    ))
                    self.buffer = ""
                    break
            else:
                # 在标签内，查找结束标签
                end_pos = self.buffer.find(self.config.closing_tag)
                if end_pos != -1:
                    # 找到结束标签
                    tag_content = self.buffer[:end_pos]
                    # 计算增量思考内容
                    delta_thinking = tag_content[len(self.last_extracted_thinking):]
                    self.last_extracted_thinking = tag_content
                    results.append(ExtractionResult(
                        content="",
                        tag_content_extracted=delta_thinking,  # 返回增量
                        is_tag_content=True,
                        complete=True
                    ))
                    # 重置状态
                    self.in_tag = False
                    self.buffer = self.buffer[end_pos + len(self.config.closing_tag):]
                    self.last_extracted_thinking = ""  # 完成后重置
                else:
                    # 没有结束标签，将当前缓冲区内容作为增量输出
                    if self.buffer:  # 只有当缓冲区不为空时才输出
                        delta_thinking = self.buffer[len(self.last_extracted_thinking):]
                        if delta_thinking:  # 只有当有新内容时才输出
                            self.last_extracted_thinking = self.buffer
                            results.append(ExtractionResult(
                                content="",
                                tag_content_extracted=delta_thinking,  # 返回增量
                                is_tag_content=True,
                                complete=False
                            ))
                    break  # 退出循环，等待更多内容
        return results

    def finalize(self) -> Optional[ExtractionResult]:
        """完成处理，处理剩余内容"""
        if self.in_tag and self.buffer:
            # 处理未闭合的标签内容
            delta_thinking = self.buffer[len(self.last_extracted_thinking):]
            if delta_thinking:
                result = ExtractionResult(
                    content="",
                    tag_content_extracted=delta_thinking,
                    is_tag_content=True,
                    complete=False
                )
                self.buffer = ""
                self.in_tag = False
                self.last_extracted_thinking = ""
                return result
        return None


class OpenAIChatCompletionChunkConverter:
    def __init__(self, chat_request: ChatAgentRequest):
        """Initialize the converter."""
        self.chat_model = chat_request.model
        # 配置标签
        self.tag_config = TagConfig(opening_tag="<think>", closing_tag="</think>")
        self.tag_extractor = TagExtractor(self.tag_config)

    def _make_json_chunk(self, content: any):
        """Helper function to format the content as a JSON chunk."""
        return json.dumps(content, ensure_ascii=False)



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


        async for response in async_response_gen:
            if response.message.role == MessageRole.ASSISTANT:
                previous_assistant_message = response.message
                # 使用 TagExtractor 处理文本
                extraction_results = self.tag_extractor.process_text(response.delta)

                if extraction_results:
                    # 有正常文本 / 思考增量内容
                    for extraction_result in extraction_results:
                        chunk = ChatCompletionChunk(
                            id=chat_id,
                            created=int(time.time()),
                            model=model,
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": chunk_id,
                                    "delta": {
                                        "content": extraction_result.content if not extraction_result.is_tag_content else "",
                                        "role": response.message.role,
                                        "tool_calls": response.message.additional_kwargs.get('tool_calls', []),
                                        "reasoning_content": extraction_result.tag_content_extracted if extraction_result.is_tag_content else "",
                                        "reasoning_completed": extraction_result.complete,
                                    },
                                    "finish_reason": "stop" if response.message.additional_kwargs.get(
                                    "STOP_FLAG") else None,
                                },
                            ],
                        )
                        yield self._make_json_chunk(chunk.model_dump(mode="json"))
                        chunk_id += 1
                else:
                    # extraction_results 为空，但 tool_calls 有值
                    tool_calls = response.message.additional_kwargs.get('tool_calls', [])
                    if tool_calls:
                        chunk = ChatCompletionChunk(
                            id=chat_id,
                            created=int(time.time()),
                            model=model,
                            object="chat.completion.chunk",
                            choices=[
                                {
                                    "index": chunk_id,
                                    "delta": {
                                        "content": "",
                                        "role": response.message.role,
                                        "tool_calls": tool_calls,
                                        "reasoning_content": "",
                                    },
                                    "finish_reason": "stop" if response.message.additional_kwargs.get(
                                    "STOP_FLAG") else None,
                                },
                            ],
                        )
                        yield self._make_json_chunk(chunk.model_dump(mode="json"))
                        chunk_id += 1

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
                chunk_id += 1
            else:
                raise ValueError(f"Unknown role: {response.message.role}")

        # 处理可能剩余的思考内容
        final_result = self.tag_extractor.finalize()
        if final_result and final_result.tag_content_extracted:
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    {
                        "index": chunk_id,
                        "delta": {
                            "content": "",
                            "role": MessageRole.ASSISTANT,
                            "tool_calls": [],
                            "reasoning_content": final_result.tag_content_extracted,
                            "reasoning_completed": final_result.complete,
                        },
                        "finish_reason": None,
                    },
                ],
            )
            yield self._make_json_chunk(chunk.model_dump(mode="json"))

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
