from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from typing import Dict, Any


class OpenAIAlikeMultiModal(OpenAIMultiModal):
    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_new_tokens is not None:
            # If max_tokens is None, don't include in the payload:
            # https://platform.openai.com/docs/api-reference/chat
            # https://platform.openai.com/docs/api-reference/completions
            base_kwargs["max_tokens"] = self.max_new_tokens
        return {**base_kwargs, **self.additional_kwargs}
