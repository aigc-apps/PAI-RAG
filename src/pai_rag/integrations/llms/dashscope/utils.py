from typing import Optional, Tuple
import dashscope.version
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
import dashscope

DEFAULT_DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_API_VERSION = ""


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
