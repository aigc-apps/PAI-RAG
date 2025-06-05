# dashscope multimodal llm的achat接口有问题，这里使用openai接口
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field
from typing import Dict, Any, Optional, Union
from llama_index.llms.openai.base import Tokenizer
from transformers import AutoTokenizer


class OpenAIAlikeMultiModal(OpenAIMultiModal):
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.model_fields["context_window"].description,
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=LLMMetadata.model_fields["is_function_calling_model"].description,
    )

    tokenizer: Union[Tokenizer, str, None] = Field(
        default=None,
        description=(
            "An instance of a tokenizer object that has an encode method, or the name"
            " of a tokenizer model from Hugging Face. If left as None, then this"
            " disables inference of max_tokens."
        ),
    )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        if isinstance(self.tokenizer, str):
            return AutoTokenizer.from_pretrained(self.tokenizer)
        return self.tokenizer

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        return {**base_kwargs, **self.additional_kwargs}
