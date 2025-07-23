from typing import Dict
from contextvars import ContextVar

request_context: ContextVar[Dict[str, str]] = ContextVar("request_context", default={})


# 在中间件中
def set_context(user_args: Dict[str, str]):
    if not user_args:
        return

    request_context.set(user_args)


def get_context() -> Dict[str, str]:
    return request_context.get() or {}
