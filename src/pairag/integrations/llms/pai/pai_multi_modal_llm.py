from typing import Any, Sequence
from llama_index.core.multi_modal_llms import MultiModalLLMMetadata
from llama_index.core.schema import ImageDocument
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr, Field
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from pairag.integrations.llms.pai.llm_utils import create_multi_modal_llm
from pairag.integrations.llms.pai.llm_config import (
    OpenAICompatibleLlmConfig,
)
from pairag.integrations.llms.pai.llm_utils import (
    merge_consecutive_messages,
)
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
)
from openinference.instrumentation.llama_index import get_current_span
from pairag.integrations.trace.base import use_current_span
from pairag.integrations.llms.pai.open_ai_alike_multi_modal import (
    OpenAIAlikeMultiModal,
)
from loguru import logger


class PaiMultiModalLlm(OpenAIAlikeMultiModal):
    _llm: Any = PrivateAttr()
    llm_config: OpenAICompatibleLlmConfig = Field(
        default=None,
        description="Llm configuration",
    )

    def __init__(self, llm_config: OpenAICompatibleLlmConfig):
        super().__init__()
        self.llm_config = llm_config
        self.model = llm_config.model

        self._llm = create_multi_modal_llm(llm_config)
        self._llm.callback_manager = Settings.callback_manager
        self.callback_manager = Settings.callback_manager
        logger.info(
            f"Created PaiMultiModalLlm with {llm_config.model} - {llm_config.base_url}"
        )

    @classmethod
    def class_name(cls) -> str:
        return "PaiMultiModalLlm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi-Modal LLM metadata."""
        return MultiModalLLMMetadata(
            model_name=self.model,
            context_window=self.llm_config.context_window,
            num_output=self.llm_config.max_tokens,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument] = [], **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for Multi-Modal LLM."""
        return self._llm.complete(
            prompt=prompt, image_documents=image_documents, **kwargs
        )

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint for Multi-Modal LLM."""
        return self._llm.stream_complete(
            prompt=prompt, image_documents=image_documents, **kwargs
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat endpoint for Multi-Modal LLM."""
        return self._llm.chat(messages=messages, **kwargs)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat endpoint for Multi-Modal LLM."""
        return self._llm.stream_chat(messages=messages, **kwargs)

    # ===== Async methods =====

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument] = [], **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""
        return await self._llm.acomplete(
            prompt=prompt, image_documents=image_documents, **kwargs
        )

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument] = [], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for Multi-Modal LLM."""
        return await self._llm.astream_complete(
            prompt=prompt, image_documents=image_documents, **kwargs
        )

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat endpoint for Multi-Modal LLM."""
        """Chat with the model."""
        is_enable_thinking = (
            self.llm_config.is_reasoning_model and self._is_enable_thinking(**kwargs)
        )

        if not self.metadata.is_chat_model:
            completion_response = await self.achat(messages, **kwargs)
            return completion_response_to_chat_response(completion_response)

        _response = await self._llm.achat(messages, **kwargs)
        if is_enable_thinking:
            _response.message.content = self._wrap_with_think_tags(
                _response.message.content, _response
            )

        return _response

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for Multi-Modal LLM."""
        kwargs["stream_options"] = kwargs.get("stream_options", {"include_usage": True})
        kwargs["temperature"] = kwargs.get("temperature", self.temperature)
        messages = merge_consecutive_messages(messages)
        if self.llm_config.is_reasoning_model:
            messages.append(ChatMessage(role="assistant", content="<think>\n"))
            logger.info(
                f"add mandatory think for reasoning models, messages: {messages}"
            )
        if not self.metadata.is_chat_model:
            completion_response = await self.astream_chat(messages, **kwargs)
            return self.async_stream_completion_response_to_chat_response(
                completion_response
            )

        filterd_messages = [
            message
            for message in messages
            if message.content or message.additional_kwargs
        ]

        return await self.async_chat_response_to_chat_response_with_think(
            filterd_messages, **kwargs
        )

    def async_stream_completion_response_to_chat_response(
        self,
        completion_response_gen: CompletionResponseAsyncGen,
    ) -> ChatResponseAsyncGen:
        """Convert a stream completion response to a stream chat response."""

        async def gen() -> ChatResponseAsyncGen:
            is_first_chunk = True
            in_reasoning = False
            reasoning_ended = False
            response_content = ""
            think_tag_added = False

            async for response in completion_response_gen:
                if self.llm_config.is_reasoning_model:
                    # 尝试从raw对象中获取reasoning_content
                    reasoning_text = ""
                    if hasattr(response, "raw") and response.raw:
                        # raw是ChatCompletionChunk对象
                        if hasattr(response.raw, "choices") and response.raw.choices:
                            delta = (
                                response.raw.choices[0].delta
                                if response.raw.choices
                                else None
                            )
                            if delta and hasattr(delta, "reasoning_content"):
                                reasoning_text = delta.reasoning_content or ""

                    # 如果有reasoning_content，处理它
                    if reasoning_text:
                        if not think_tag_added:
                            # 第一次遇到reasoning_content，输出<think>标签
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="<think>",
                                    additional_kwargs=response.additional_kwargs,
                                ),
                                delta="<think>",
                                raw="<think>",
                            )
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="<think>\n",
                                    additional_kwargs=response.additional_kwargs,
                                ),
                                delta="\n",
                                raw="\n",
                            )
                            response_content = "<think>\n"
                            think_tag_added = True
                            in_reasoning = True

                        # 输出reasoning_content内容（每个片段都要输出）
                        response_content += reasoning_text
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=response_content,
                                additional_kwargs=response.additional_kwargs,
                            ),
                            delta=reasoning_text,
                            raw=response.raw,
                        )
                        # 继续下一个循环，不处理delta
                        continue

                    # 检查reasoning是否结束（通过delta内容判断）
                    if in_reasoning and not reasoning_ended and response.delta:
                        # 如果有delta内容且之前在reasoning中，说明reasoning结束了
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=response_content + "\n</think>",
                                additional_kwargs=response.additional_kwargs,
                            ),
                            delta="\n</think>",
                            raw="\n</think>",
                        )
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=response_content + "\n</think>\n\n",
                                additional_kwargs=response.additional_kwargs,
                            ),
                            delta="\n\n",
                            raw="\n\n",
                        )
                        response_content += "\n</think>\n\n"
                        reasoning_ended = True
                        in_reasoning = False

                    # 处理第一个chunk，如果没有reasoning_content但需要添加<think>
                    if is_first_chunk and response.delta:
                        if not think_tag_added and not response.text.startswith(
                            "<think>"
                        ):
                            # 没有reasoning_content，但需要补充<think>
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="<think>",
                                    additional_kwargs=response.additional_kwargs,
                                ),
                                delta="<think>",
                                raw="<think>",
                            )
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="<think>\n",
                                    additional_kwargs=response.additional_kwargs,
                                ),
                                delta="\n",
                                raw="\n",
                            )
                            response_content = "<think>\n"
                            think_tag_added = True
                        is_first_chunk = False

                # 输出正常的delta内容
                if response.delta:
                    response_content += response.delta
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=response_content,
                            additional_kwargs=response.additional_kwargs,
                        ),
                        delta=response.delta,
                        raw=response.raw,
                    )

        return gen()

    async def async_chat_response_to_chat_response_with_think(
        self, messages, **kwargs
    ) -> ChatResponseAsyncGen:
        is_enable_thinking = (
            self.llm_config.is_reasoning_model and self._is_enable_thinking(**kwargs)
        )
        if is_enable_thinking:
            logger.info("Using reasoning models with think.")
        if not is_enable_thinking:

            @use_current_span(get_current_span())
            async def gen() -> ChatResponseAsyncGen:
                async for response in await self._llm.astream_chat(messages, **kwargs):
                    yield response

            return gen()
        else:

            @use_current_span(get_current_span())
            async def gen() -> ChatResponseAsyncGen:
                is_first_chunk = True
                in_reasoning = False
                reasoning_ended = False
                response_content = ""
                think_tag_added = False

                async for response in await self._llm.astream_chat(messages, **kwargs):
                    # 尝试从raw对象中获取reasoning_content
                    reasoning_text = ""
                    if hasattr(response, "raw") and response.raw:
                        # raw是ChatCompletionChunk对象
                        if hasattr(response.raw, "choices") and response.raw.choices:
                            delta = (
                                response.raw.choices[0].delta
                                if response.raw.choices
                                else None
                            )
                            if delta and hasattr(delta, "reasoning_content"):
                                reasoning_text = delta.reasoning_content or ""

                    # 如果有reasoning_content，处理它
                    if reasoning_text:
                        if not think_tag_added:
                            # 第一次遇到reasoning_content，输出<think>标签
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="<think>",
                                ),
                                delta="<think>",
                            )
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="<think>\n",
                                ),
                                delta="\n",
                            )
                            response_content = "<think>\n"
                            think_tag_added = True
                            in_reasoning = True

                        # 输出reasoning_content内容（每个片段都要输出）
                        response_content += reasoning_text
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=response_content,
                                additional_kwargs=response.additional_kwargs,
                            ),
                            delta=reasoning_text,
                            raw=response.raw,
                        )
                        # 继续下一个循环，不处理delta
                        continue

                    # 检查reasoning是否结束（通过delta内容判断）
                    if in_reasoning and not reasoning_ended and response.delta:
                        # 如果有delta内容且之前在reasoning中，说明reasoning结束了
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=response_content + "\n</think>",
                            ),
                            delta="\n</think>",
                        )
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=response_content + "\n</think>\n\n",
                            ),
                            delta="\n\n",
                        )
                        response_content += "\n</think>\n\n"
                        reasoning_ended = True
                        in_reasoning = False

                    # 处理第一个chunk，如果没有reasoning_content但需要添加<think>
                    if is_first_chunk and response.delta:
                        if not think_tag_added and not response.delta.startswith(
                            "<think>"
                        ):
                            # 没有reasoning_content，但需要补充<think>
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="<think>",
                                ),
                                delta="<think>",
                            )
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="<think>\n",
                                ),
                                delta="\n",
                            )
                            response_content = "<think>\n"
                            think_tag_added = True
                        is_first_chunk = False

                    # 输出正常的delta内容
                    if response.delta:
                        response_content += response.delta
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=response_content,
                            ),
                            delta=response.delta,
                        )

        return gen()

    def _is_enable_thinking(self, **kwargs) -> bool:
        """是否打开think开关，默认为True，对于Qwen3系列，可设置为False"""
        enable_thinking = kwargs.get(
            "extra_body",
            {},
        ).get("enable_thinking", True)

        return enable_thinking

    def _wrap_with_think_tags(self, content: str, response: Any = None) -> str:
        """处理reasoning_content并添加<think>标签封装"""
        reasoning_content = ""

        # 尝试从response的raw对象中获取reasoning_content
        if response and hasattr(response, "raw") and response.raw:
            # raw是ChatCompletion对象
            if hasattr(response.raw, "choices") and response.raw.choices:
                choice = response.raw.choices[0] if response.raw.choices else None
                if (
                    choice
                    and hasattr(choice, "message")
                    and hasattr(choice.message, "reasoning_content")
                ):
                    reasoning_content = choice.message.reasoning_content or ""

        if reasoning_content:
            # 如果有reasoning_content，封装成<think>reasoning_content</think> + content
            if content:
                return f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            else:
                return f"<think>\n{reasoning_content}\n</think>"
        else:
            # 如果没有reasoning_content，检查是否需要补充<think>开头
            if content and not content.startswith("<think>"):
                return f"<think>\n{content}"
            return content
