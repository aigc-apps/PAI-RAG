from utils.constants import try_get_int_env
from enum import Enum


MAX_CHAT_STEPS = try_get_int_env("MAX_CHAT_STEPS", 15)

DEFAULT_MAX_INPUT_TOKENS = 22000
DEFAULT_HISTORY_MESSAGES_COUNT = 50
DEFAULT_HISTORY_MESSAGES_INPUT_TOKENS = 1500
DEFAULT_AGENT_HISTORY_ROUNDS = 5


class MessageRole(str, Enum):
    TOOL = "tool"
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
