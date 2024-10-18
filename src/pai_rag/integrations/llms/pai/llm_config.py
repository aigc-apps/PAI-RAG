from typing import Literal
from pydantic import BaseModel, field_validator
from enum import Enum
from llama_index.core.constants import DEFAULT_TEMPERATURE

DEFAULT_MAX_TOKENS = 2000


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
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_PLUS: {
        "context_window": 1024 * 32,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_MAX: {
        "context_window": 1024 * 8,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_MAX_1201: {
        "context_window": 1024 * 8,
        "num_output": 1024 * 2,
        "is_chat_model": True,
        "is_function_calling_model": True,
    },
    DashScopeGenerationModels.QWEN_MAX_LONGCONTEXT: {
        "context_window": 1024 * 30,
        "num_output": 1024 * 2,
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
    paieas = "paieas"


class PaiBaseLlmConfig(BaseModel):
    source: SupportedLlmType
    temperature: float = DEFAULT_TEMPERATURE
    system_prompt: str = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    model: str = None

    @classmethod
    def get_subclasses(cls):
        return tuple(cls.__subclasses__())

    class Config:
        frozen = True

    @classmethod
    def get_type(cls):
        return cls.model_fields["source"].default

    @field_validator("source", mode="before")
    def validate_case_insensitive(cls, value):
        if isinstance(value, str):
            return value.lower()
        return value


class DashScopeLlmConfig(PaiBaseLlmConfig):
    source: Literal[SupportedLlmType.dashscope] = SupportedLlmType.dashscope
    api_key: str | None = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-turbo"


class OpenAILlmConfig(PaiBaseLlmConfig):
    source: Literal[SupportedLlmType.openai] = SupportedLlmType.openai
    api_key: str | None = None
    model: str = "gpt-3.5-turbo"


class PaiEasLlmConfig(PaiBaseLlmConfig):
    source: Literal[SupportedLlmType.paieas] = SupportedLlmType.paieas
    endpoint: str
    token: str
    model: str = "default"


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
