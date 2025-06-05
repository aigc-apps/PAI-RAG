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
from pairag.integrations.llms.utils.utils import transform_to_image_nodes
from loguru import logger


class PaiMultiModalLlm(OpenAIAlikeMultiModal):
    _llm: Any = PrivateAttr()
    llm_config: OpenAICompatibleLlmConfig = Field(
        default=None,
        description="Llm configuration",
    )

    def messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
        """Convert messages to a prompt string."""
        string_messages = []
        for message in messages:
            role = message.role
            content = message.content
            string_message = f"{role.value}: {content}"

            additional_kwargs = message.additional_kwargs
            if additional_kwargs:
                string_message += f"\n{additional_kwargs}"
            string_messages.append(string_message)

        string_messages.append(f"{MessageRole.ASSISTANT.value}: ")
        return "\n".join(string_messages)

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
        image_documents: Sequence[ImageDocument] = [],
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat endpoint for Multi-Modal LLM."""
        query_str = messages[-1].content
        additonal_images = transform_to_image_nodes(query_str)
        if additonal_images:
            image_documents.extend(additonal_images)
        prompt = self.messages_to_prompt(messages)
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER,
            image_documents=image_documents,
        )
        return self._llm.chat(messages=[chat_message], **kwargs)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Sequence[ImageDocument] = [],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat endpoint for Multi-Modal LLM."""
        query_str = messages[-1].content
        additonal_images = transform_to_image_nodes(query_str)
        if additonal_images:
            image_documents.extend(additonal_images)
        prompt = self.messages_to_prompt(messages)
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER,
            image_documents=image_documents,
        )
        return self._llm.stream_chat(messages=[chat_message], **kwargs)

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
        image_documents: Sequence[ImageDocument] = [],
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat endpoint for Multi-Modal LLM."""
        """Chat with the model."""
        query_str = messages[-1].content
        additonal_images = transform_to_image_nodes(query_str)
        if additonal_images:
            image_documents.extend(additonal_images)
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = self.complete(prompt, image_documents, **kwargs)
            return completion_response_to_chat_response(completion_response)

        prompt = self.messages_to_prompt(messages)
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER,
            image_documents=image_documents,
        )

        return self._llm.chat([chat_message], **kwargs)

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Sequence[ImageDocument] = [],
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
        query_str = messages[-1].content
        additonal_images = transform_to_image_nodes(query_str)
        if additonal_images:
            image_documents.extend(additonal_images)
        logger.info(f"images: {image_documents}")
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = await self.astream_complete(
                prompt, image_documents, **kwargs
            )
            return self.async_stream_completion_response_to_chat_response(
                completion_response
            )

        filterd_messages = [
            message
            for message in messages
            if message.content or message.additional_kwargs
        ]

        return await self.async_chat_response_to_chat_response_with_think(
            filterd_messages, image_documents, **kwargs
        )
        # return await self._llm.astream_chat(messages=messages, **kwargs)

    def async_stream_completion_response_to_chat_response(
        self,
        completion_response_gen: CompletionResponseAsyncGen,
    ) -> ChatResponseAsyncGen:
        """Convert a stream completion response to a stream chat response."""

        async def gen() -> ChatResponseAsyncGen:
            start_label = True
            async for response in completion_response_gen:
                if self.llm_config.is_reasoning_model:
                    if start_label and not response.text.startswith("<think>"):
                        start_label = False
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
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response.text,
                        additional_kwargs=response.additional_kwargs,
                    ),
                    delta=response.delta,
                    raw=response.raw,
                )

        return gen()

    async def async_chat_response_to_chat_response_with_think(
        self, messages, image_documents, **kwargs
    ) -> ChatResponseAsyncGen:
        is_enable_thinking = (
            self.llm_config.is_reasoning_model and self._is_enable_thinking(**kwargs)
        )
        prompt = self.messages_to_prompt(messages)
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER,
            image_documents=image_documents,
        )
        if is_enable_thinking:
            logger.info("Using reasoning models with think.")
        if not is_enable_thinking:

            @use_current_span(get_current_span())
            async def gen() -> ChatResponseAsyncGen:
                async for response in await self._llm.astream_chat(
                    [chat_message], **kwargs
                ):
                    yield response

            return gen()
        else:

            @use_current_span(get_current_span())
            async def gen() -> ChatResponseAsyncGen:
                start_label = True
                async for response in await self._llm.astream_chat(
                    [chat_message], **kwargs
                ):
                    if start_label and not response.delta:
                        continue
                    if start_label and not response.delta.startswith("<think>"):
                        start_label = False
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
                    else:
                        start_label = False

                    yield response

        return gen()
