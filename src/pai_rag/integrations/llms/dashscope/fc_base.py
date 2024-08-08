from typing import Any, Dict, Optional, Tuple
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
import dashscope
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.openai_like import OpenAILike


class DashScopeGenerationModels:
    """DashScope Qwen serial models."""

    QWEN_TURBO = "qwen-turbo"
    QWEN_PLUS = "qwen-plus"
    QWEN_MAX = "qwen-max"
    QWEN_MAX_1201 = "qwen-max-1201"
    QWEN_MAX_LONGCONTEXT = "qwen-max-longcontext"

    QWEM1P5_1P8B_CHAT = "qwen1.5-1.8b-chat"
    QWEM1P5_7B_CHAT = "qwen1.5-7b-chat"
    QWEM1P5_14B_CHAT = "qwen1.5-14b-chat"
    QWEM1P5_32B_CHAT = "qwen1.5-32b-chat"
    QWEM1P5_72B_CHAT = "qwen1.5-72b-chat"
    QWEM1P5_110B_CHAT = "qwen1.5-110b-chat"

    QWEM2_1P5B_INSTRUCT = "qwen2-1.5b-instruct"
    QWEM2_7B_INSTRUCT = "qwen2-7b-instruct"
    QWEM2_72B_INSTRUCT = "qwen2-72b-instruct"


DASHSCOPE_MODEL_META = {
    DashScopeGenerationModels.QWEN_TURBO: {
        "context_window": 1024 * 8,
        "num_output": 1024 * 8,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_PLUS: {
        "context_window": 1024 * 32,
        "num_output": 1024 * 32,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_MAX: {
        "context_window": 1024 * 8,
        "num_output": 1024 * 8,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_MAX_1201: {
        "context_window": 1024 * 8,
        "num_output": 1024 * 8,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_MAX_LONGCONTEXT: {
        "context_window": 1024 * 30,
        "num_output": 1024 * 30,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_1P8B_CHAT: {
        "context_window": 1024 * 30,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_7B_CHAT: {
        "context_window": 1024 * 8,
        "num_output": 1024 * 8,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_14B_CHAT: {
        "context_window": 1024 * 16,
        "num_output": 1024 * 16,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_32B_CHAT: {
        "context_window": 1024 * 16,
        "num_output": 1024 * 16,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_72B_CHAT: {
        "context_window": 1024 * 16,
        "num_output": 1024 * 16,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_110B_CHAT: {
        "context_window": 1024 * 32,
        "num_output": 1024 * 32,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM2_1P5B_INSTRUCT: {
        "context_window": 1024 * 30,
        "num_output": 1024 * 32,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM2_7B_INSTRUCT: {
        "context_window": 1024 * 32,
        "num_output": 1024 * 32,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM2_72B_INSTRUCT: {
        "context_window": 1024 * 32,
        "num_output": 1024 * 32,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
}

DEFAULT_DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def resolve_dashscope_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """ "Resolve OpenAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "DASHSCOPE_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "DASHSCOPE_API_BASE", "")

    # resolve from openai module or default
    final_api_key = api_key or dashscope.api_key or ""
    final_api_base = api_base or DEFAULT_DASHSCOPE_API_BASE

    return final_api_key, str(final_api_base)


class MyFCDashScope(OpenAILike):
    """
    MyFCDashScope LLM is a thin wrapper around the OpenAILike model that makes it compatible
    with Function Calling DashScope.
    """

    def __init__(
        self,
        model: Optional[str] = DashScopeGenerationModels.QWEN_MAX,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        api_key, api_base = resolve_dashscope_credentials(
            api_key=api_key,
            api_base=api_base,
        )

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "fc_dashscope_llm"

    @property
    def metadata(self) -> LLMMetadata:
        DASHSCOPE_MODEL_META[self.model]["num_output"] = (
            self.max_tokens or DASHSCOPE_MODEL_META[self.model]["num_output"]
        )
        return LLMMetadata(
            model_name=self.model,
            **DASHSCOPE_MODEL_META[self.model],
        )
