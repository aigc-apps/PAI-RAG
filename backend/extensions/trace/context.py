# context.py
import contextvars
from loguru import logger

# indexed in arms for reference
USER_ID = "gen_ai.user.id"
USER_NAME = "gen_ai.user.name"
SESSION_ID = "gen_ai.session.id"

# baggage key for request_id from agentscope
AGENTSCOPE_REQUEST_ID_KEY = "traffic.llm_sdk.agentscope.request_id"

# ContextVar for request_id
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


custom_context_vars = {}


def init_custom_context(keys):
    for k in keys:
        custom_context_vars[k] = contextvars.ContextVar(k, default=None)


def get_context_vars():
    return {k: v.get() for k, v in custom_context_vars.items()}


def set_context_var(key, value):
    if key in custom_context_vars:
        custom_context_vars[key].set(value)
    else:
        logger.warning(f"key: `{key}` is not in custom_context_vars")


def set_request_id(request_id: str | None):
    """Set the request_id in the current context."""
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    """Get the request_id from the current context."""
    request_id = _request_id_var.get()

    return request_id
