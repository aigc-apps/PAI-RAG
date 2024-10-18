from typing import Any, Sequence
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
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
    async_stream_completion_response_to_chat_response,
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from pai_rag.integrations.llms.pai.llm_utils import create_llm
from pai_rag.integrations.llms.pai.llm_config import (
    DASHSCOPE_MODEL_META,
    DEFAULT_MAX_TOKENS,
    PaiBaseLlmConfig,
)
import logging

logger = logging.getLogger(__name__)


class PaiLlm(OpenAILike):
    _llm: Any = PrivateAttr()

    def __init__(self, llm_config: PaiBaseLlmConfig):
        super().__init__()
        self._llm = create_llm(llm_config)
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
                num_output=DEFAULT_MAX_TOKENS,
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
        """Chat with the model."""
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = await self.acomplete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion_response)

        return await self._llm.achat(messages, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = await self.astream_complete(
                prompt, formatted=True, **kwargs
            )
            return async_stream_completion_response_to_chat_response(
                completion_response
            )

        return await self._llm.astream_chat(messages, **kwargs)
