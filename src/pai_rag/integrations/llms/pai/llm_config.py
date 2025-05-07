import os
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from llama_index.core.constants import DEFAULT_TEMPERATURE

DEFAULT_CONTEXT_WINDOW = 8000
DEFAULT_MAX_TOKENS = 4000
DEFAULT_MLLM_MAX_TOKENS = 2048


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
    dashscope = "dashscope"
    openai = "openai"
    openai_compatible = "openai_compatible"
    paieas = "paieas"


class PaiBaseLlmConfig(BaseModel):
    source: SupportedLlmType | None = None
    temperature: float = DEFAULT_TEMPERATURE
    system_prompt: str | None = None
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    vision_support: bool | None = None
    is_reasoning_model: bool | None = None
    is_streaming_model: bool | None = None
    model_id: str | None = None

    model_config = ConfigDict(coerce_numbers_to_str=True, frozen=False)

    @classmethod
    def get_subclasses(cls):
        return tuple(cls.__subclasses__())

    @classmethod
    def get_type(cls):
        return cls.model_fields["source"].default

    def is_validate(self):
        return all(
            [
                self.source not in [None, ""],
                self.base_url not in [None, ""],
                self.model not in [None, ""],
            ]
        )


class DashScopeLlmConfig(PaiBaseLlmConfig):
    source: Literal[SupportedLlmType.dashscope] = SupportedLlmType.dashscope
    api_key: str | None = Field(default=os.getenv("DASHSCOPE_API_KEY"))  # use default
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-max"


class OpenAILlmConfig(PaiBaseLlmConfig):
    source: Literal[SupportedLlmType.openai] = SupportedLlmType.openai
    api_key: str | None = None
    model: str = "gpt-3.5-turbo"


class OpenAICompatibleLlmConfig(PaiBaseLlmConfig):
    source: Literal[
        SupportedLlmType.openai_compatible
    ] = SupportedLlmType.openai_compatible
    base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str | None = None
    model: str = ""


class PaiEasLlmConfig(PaiBaseLlmConfig):
    source: Literal[SupportedLlmType.paieas] = SupportedLlmType.paieas
    endpoint: str
    token: str
    model: str = "default"


class DashScopeMultiModalLlmConfig(DashScopeLlmConfig):
    model: str = "qwen-vl-max"


SupporttedLlmClsMap = {cls.get_type(): cls for cls in PaiBaseLlmConfig.get_subclasses()}


def parse_llm_config(config_data):
    if "source" not in config_data:
        raise ValueError("Llm config must contain 'source' field")

    llm_cls = SupporttedLlmClsMap.get(config_data["source"].lower())
    if llm_cls is None:
        raise ValueError(f"Unsupported llm source: {config_data['source']}")

    return llm_cls(**config_data)


if __name__ == "__main__":
    llm_config_data = {
        "source": "dashscope",
        "model": "qwen-turbo",
        "api_key": None,
        "max_tokens": 1024,
    }
    print(parse_llm_config(llm_config_data))
