from typing import Any, Sequence
from llama_index.core.multi_modal_llms import MultiModalLLM, MultiModalLLMMetadata
from llama_index.core.schema import ImageDocument
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from pai_rag.integrations.llms.pai.llm_utils import create_multi_modal_llm
from pai_rag.integrations.llms.pai.llm_config import PaiBaseLlmConfig


class PaiMultiModalLlm(MultiModalLLM):
    _llm: MultiModalLLM = PrivateAttr()

    def __init__(self, llm_config: PaiBaseLlmConfig):
        super().__init__()

        self._llm = create_multi_modal_llm(llm_config)
        self._llm.callback_manager = Settings.callback_manager
        self.callback_manager = Settings.callback_manager

    @classmethod
    def class_name(cls) -> str:
        return "PaiMultiModalLlm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi-Modal LLM metadata."""
        return self._llm.metadata

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
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
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""
        return await self._llm.acomplete(
            prompt=prompt, image_documents=image_documents, **kwargs
        )

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
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
        return await self._llm.achat(messages=messages, **kwargs)

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for Multi-Modal LLM."""
        return await self._llm.astream_chat(messages=messages, **kwargs)
