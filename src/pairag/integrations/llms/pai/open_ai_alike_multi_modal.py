from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.llms.openai_like import OpenAILike
from typing import Dict, Any


class OpenAIAlikeMultiModal(OpenAIMultiModal, OpenAILike):
    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        return {**base_kwargs, **self.additional_kwargs}
