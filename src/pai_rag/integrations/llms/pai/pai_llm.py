from typing import Any, Sequence
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr, Field
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from pai_rag.integrations.llms.pai.llm_utils import (
    create_llm,
    merge_consecutive_messages,
)
from pai_rag.integrations.llms.pai.llm_config import (
    DASHSCOPE_MODEL_META,
    PaiBaseLlmConfig,
)
from llama_index.core.base.llms.types import MessageRole
import llama_index.core.instrumentation as instrument

from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)

dispatcher = instrument.get_dispatcher(__name__)

from loguru import logger


class PaiLlm(OpenAILike):
    _llm: Any = PrivateAttr()
    llm_config: PaiBaseLlmConfig = Field(
        default=None,
        description="Llm configuration",
    )

    def __init__(self, llm_config: PaiBaseLlmConfig):
        super().__init__(
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )
        self.llm_config = llm_config
        self._llm = create_llm(self.llm_config)
        self.model = llm_config.model
        self._llm.callback_manager = Settings.callback_manager
        self.callback_manager = Settings.callback_manager

    @classmethod
    def class_name(cls) -> str:
        return "PaiLlm"

    @property
    def metadata(self) -> LLMMetadata:
        if self.model in DASHSCOPE_MODEL_META:
            return LLMMetadata(
                model_name=self.model,
                **DASHSCOPE_MODEL_META[self.model],
            )
        else:
            return LLMMetadata(
                model_name=self.model,
                num_output=self.llm_config.max_tokens,
                is_chat_model=True,
                is_function_calling_model=True,
            )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt."""
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return self._llm.complete(prompt, **kwargs)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream complete the prompt."""
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return self._llm.stream_complete(prompt, **kwargs)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the model."""
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = self.complete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion_response)

        return self._llm.chat(messages, **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
            return stream_completion_response_to_chat_response(completion_response)

        return self._llm.stream_chat(messages, **kwargs)

    # -- Async methods --

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt."""
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return await self._llm.acomplete(prompt, **kwargs)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Stream complete the prompt."""
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        return await self._llm.astream_complete(prompt, **kwargs)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        dispatcher.event(
            LLMChatStartEvent(
                messages=messages,
                additional_kwargs=kwargs,
                model_dict={}
            )
        )
        messages = merge_consecutive_messages(messages)
        kwargs["temperature"] = kwargs.get("temperature", self.temperature)
        kwargs["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
        # add mandatory think for reasoning models
        if self.llm_config.is_reasoning_model:
            messages.append(ChatMessage(role="assistant", content="<think>\n"))
            logger.info(
                f"add mandatory think for reasoning models, messages: {messages}"
            )
        """Chat with the model."""
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            logger.info(f"llm complete, prompt: {prompt}")
            completion_response = await self.acomplete(prompt, formatted=True, **kwargs)
            if self.llm_config.is_reasoning_model and not str(
                completion_response.text
            ).startswith("<think>"):
                completion_response.text = "<think>\n" + completion_response.text
            return completion_response_to_chat_response(completion_response)

        filterd_messages = [
            message
            for message in messages
            if message.content or message.additional_kwargs
        ]
        logger.info(f"llm chat, filterd_messages: {filterd_messages}")
        _response = await self._llm.achat(filterd_messages, **kwargs)
        if self.llm_config.is_reasoning_model and not str(_response.delta).startswith(
            "<think>"
        ):
            _response.message.content = "<think>\n" + _response.message.content

        dispatcher.event(
            LLMChatEndEvent(
                messages=messages,
                response=_response
            )
        )

        return _response

    def async_stream_completion_response_to_chat_response(
        self,
        completion_response_gen: CompletionResponseAsyncGen,
    ) -> ChatResponseAsyncGen:
        """Convert a stream completion response to a stream chat response."""

        async def gen() -> ChatResponseAsyncGen:
            start_label = True
            response_content = ""
            additional_kwargs = {}
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
                        response_content = "<think>\n"

                response_content += response.delta
                additional_kwargs = response.additional_kwargs

                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response_content,
                        additional_kwargs=response.additional_kwargs,
                    ),
                    delta=response.delta,
                    raw=response.raw,
                )
            """
            result_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_content,
                additional_kwargs=additional_kwargs,
            )
            print("dispatching llm end event")

            dispatcher.event(LLMChatEndEvent(
                messages=[],
                response=ChatResponse(
                    message=result_msg
                )))
            print("dispatched llm end event")
            """
        return gen()

    async def async_chat_response_to_chat_response_with_think(
        self, messages, **kwargs
    ) -> ChatResponseAsyncGen:
        response_iter_async = await self._llm.astream_chat(messages, **kwargs)
        
        async def gen() -> ChatResponseAsyncGen:
            response_content = ""
            additional_kwargs = {}
            start_label = True
            async for response in response_iter_async:
                if self.llm_config.is_reasoning_model and start_label and not response.delta.startswith("<think>"):
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
                    response_content = "<think>\n"

                response_content += response.delta
                additional_kwargs = response.additional_kwargs
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response_content,
                    ),
                    delta = response.delta,
                    additional_kwargs=response.additional_kwargs,
                )

            result_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_content,
                additional_kwargs=additional_kwargs,
            )
            """
            print("dispatching llm end event")

            dispatcher.event(LLMChatEndEvent(
                messages=[],
                response=ChatResponse(
                    message=result_msg
                )))
            print("dispatched llm end event")
            """
        return gen()

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """
        dispatcher.event(
            LLMChatStartEvent(
                messages=messages,
                additional_kwargs=kwargs,
                model_dict={}
            )
        )
        """
        kwargs["stream_options"] = kwargs.get("stream_options", {"include_usage": True})
        kwargs["temperature"] = kwargs.get("temperature", self.temperature)
        kwargs["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
        messages = merge_consecutive_messages(messages)
        if self.llm_config.is_reasoning_model:
            messages.append(ChatMessage(role="assistant", content="<think>\n"))
            logger.info(
                f"add mandatory think for reasoning models, messages: {messages}"
            )
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = await self.astream_complete(
                prompt, formatted=True, **kwargs
            )
            response_gen = self.async_stream_completion_response_to_chat_response(
                completion_response
            )
            return response_gen

        filterd_messages = [
            message
            for message in messages
            if message.content or message.additional_kwargs
        ]

        response_gen = await self.async_chat_response_to_chat_response_with_think(
            filterd_messages, **kwargs
        )
        return response_gen
