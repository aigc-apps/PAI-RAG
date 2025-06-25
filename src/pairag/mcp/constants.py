from pairag.utils.constants import try_get_int_env


MAX_CHAT_STEPS = try_get_int_env("MAX_CHAT_STEPS", 15)

DEFAULT_MAX_INPUT_TOKENS = 32768
DEFAULT_SUMMARIZER_LENGTH = 5000
