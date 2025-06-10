import json
import os
from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator
from enum import Enum
from llama_index.core.constants import DEFAULT_TEMPERATURE

DEFAULT_CONTEXT_WINDOW = 8000
DEFAULT_MAX_TOKENS = 4000
DEFAULT_MLLM_MAX_TOKENS = 2048
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Compatible with EAS
EAS_LLM_ENDPOINT_VARIABLE_NAME = "PAIRAG_RAG__LLM__endpoint"
EAS_LLM_TOKEN_VARIABLE_NAME = "PAIRAG_RAG__LLM__token"


class DashScopeGenerationModels:
    """DashScope Qwen serial models."""

    QWEN_TURBO = "qwen-turbo"
    QWEN_PLUS = "qwen-plus"
    QWEN_MAX = "qwen-max"
    QWEN_LONG = "qwen-long"

    QWEN2P5_7B_INSTRUCT = "qwen2.5-7b-instruct"
    QWEN2P5_14B_INSTRUCT = "qwen2.5-14b-instruct"
    QWEN2P5_32B_INSTRUCT = "qwen2.5-32b-instruct"
    QWEN2P5_72B_INSTRUCT = "qwen2.5-72b-instruct"

    DEEPSEEK_R1_671B = "deepseek-r1"
    DEEPSEEK_V3_671B = "deepseek-v3"
    DEEPSEEK_R1_DISTILL_QWEN_7B = "deepseek-r1-distill-qwen-7b"
    DEEPSEEK_R1_DISTILL_QWEN_14B = "deepseek-r1-distill-qwen-14b"
    DEEPSEEK_R1_DISTILL_QWEN_32B = "deepseek-r1-distill-qwen-32b"
    DEEPSEEK_R1_DISTILL_LLAMA_8B = "deepseek-r1-distill-llama-8b"
    DEEPSEEK_R1_DISTILL_LLAMA_70B = "deepseek-r1-distill-llama-70b"

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
        "context_window": 1000000,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_PLUS: {
        "context_window": 131072,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_MAX: {
        "context_window": 32768,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_LONG: {
        "context_window": 10000000,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN2P5_7B_INSTRUCT: {
        "context_window": 131072,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN2P5_14B_INSTRUCT: {
        "context_window": 131072,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN2P5_32B_INSTRUCT: {
        "context_window": 131072,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN2P5_72B_INSTRUCT: {
        "context_window": 131072,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.DEEPSEEK_R1_671B: {
        "context_window": 65792,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": False,
    },
    DashScopeGenerationModels.DEEPSEEK_V3_671B: {
        "context_window": 65792,
        "num_output": 8192,
        "is_chat_model": True,
        "is_function_calling_model": False,
    },
    DashScopeGenerationModels.DEEPSEEK_R1_DISTILL_QWEN_7B: {
        "context_window": 32768,
        "num_output": 16384,
        "is_chat_model": True,
        "is_function_calling_model": False,
    },
    DashScopeGenerationModels.DEEPSEEK_R1_DISTILL_QWEN_14B: {
        "context_window": 32768,
        "num_output": 16384,
        "is_chat_model": True,
        "is_function_calling_model": False,
    },
    DashScopeGenerationModels.DEEPSEEK_R1_DISTILL_QWEN_32B: {
        "context_window": 32768,
        "num_output": 16384,
        "is_chat_model": True,
        "is_function_calling_model": False,
    },
    DashScopeGenerationModels.DEEPSEEK_R1_DISTILL_LLAMA_8B: {
        "context_window": 32768,
        "num_output": 16384,
        "is_chat_model": True,
        "is_function_calling_model": False,
    },
    DashScopeGenerationModels.DEEPSEEK_R1_DISTILL_LLAMA_70B: {
        "context_window": 32768,
        "num_output": 16384,
        "is_chat_model": True,
        "is_function_calling_model": False,
    },
    DashScopeGenerationModels.QWEM1P5_1P8B_CHAT: {
        "context_window": 1024 * 30,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_7B_CHAT: {
        "context_window": 1024 * 8,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_14B_CHAT: {
        "context_window": 1024 * 16,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_32B_CHAT: {
        "context_window": 1024 * 16,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_72B_CHAT: {
        "context_window": 1024 * 16,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM1P5_110B_CHAT: {
        "context_window": 1024 * 32,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM2_1P5B_INSTRUCT: {
        "context_window": 1024 * 30,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM2_7B_INSTRUCT: {
        "context_window": 1024 * 32,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEM2_72B_INSTRUCT: {
        "context_window": 1024 * 32,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
}


class SupportedLlmType(str, Enum):
    openai_compatible = "openai_compatible"


class OpenAICompatibleLlmConfig(BaseModel):
    source: Literal[
        SupportedLlmType.openai_compatible
    ] = SupportedLlmType.openai_compatible
    base_url: str | None = os.environ.get(
        EAS_LLM_ENDPOINT_VARIABLE_NAME, DASHSCOPE_BASE_URL
    )
    api_key: str | None = os.environ.get(EAS_LLM_TOKEN_VARIABLE_NAME, "")
    model: str = ""

    temperature: float = DEFAULT_TEMPERATURE
    system_prompt: str | None = None
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    vision_support: bool | None = None
    is_reasoning_model: bool | None = None  # reasoning support
    is_streaming_only: bool | None = None  # only supports streaming mode
    model_id: str | None = "default"  # unique model id
    extra_body_str: str | None = None  # extra body params

    model_config = ConfigDict(coerce_numbers_to_str=True, frozen=False)

    def is_validate(self):
        return all(
            [
                self.source not in [None, ""],
                self.base_url not in [None, ""],
                self.model not in [None, ""],
            ]
        )

    @model_validator(mode="before")
    def normalize_data(cls, data: dict) -> dict:
        # Convert name to title case
        if "extra_body_str" in data and isinstance(data["extra_body_str"], dict):
            data["extra_body_str"] = json.dumps(data["extra_body_str"])

        return data
