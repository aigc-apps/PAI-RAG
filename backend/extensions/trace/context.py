# context.py
import contextvars
from loguru import logger

# indexed in arms for reference
USER_ID = "gen_ai.user.id"
USER_NAME = "gen_ai.user.name"
SESSION_ID = "gen_ai.session.id"


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
