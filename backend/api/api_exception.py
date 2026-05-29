import functools
import traceback
from typing import Any, Awaitable, Callable, Optional, TypeVar

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ValidationError

from common.chat.response_model import to_dict


# --- 1. 定义统一的错误响应 Schema ---
class ApiExceptionResponse(BaseModel):
    """用于文档和类型提示的统一错误响应格式"""
    code: int
    message: str
    data: Optional[Any] = None

# --- 2. 定义ApiException类 ---
class ApiException(HTTPException):
    """继承 HTTPException，并添加 data 字段"""

    # 注意：status_code 和 detail 是继承自 HTTPException 的
    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        super().__init__(status_code=code, detail=message)
        self.data = data

    # 可选：为了方便使用，提供一个 classmethod
    @classmethod
    def not_found(cls, resource_id: str, resource_type: str):
        return cls(
            code=404,
            message=f"{resource_type} with ID {resource_id} not found.",
            data={"resource": resource_type, "id": resource_id}
        )


async def api_exception_handler(request: Request, exc: ApiException):
    """
    捕获 ApiException，并将其格式化为 ApiExceptionResponse 结构。
    """
    # 构造自定义的 JSON 响应体，利用 CustomAPIException 携带的 data 属性
    response_data = {
        "code": exc.status_code,
        "message": exc.detail,
        "data": to_dict(exc.data)
    }

    # 返回 JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=exc.headers
    )


# ---------------------------------------------------------------------------
# Unified router-level error handling
# ---------------------------------------------------------------------------
# Routers across the codebase repeat the same try/except scaffolding:
#
#     try:
#         result = await svc.do_something(...)
#         return success_response(...)
#     except ValueError as e:
#         logger.error(...)
#         raise ApiException(code=400, message=str(e))
#     except Exception as e:
#         logger.error(...)
#         raise ApiException(code=500, message=f"Failed to ...: {e}")
#
# This module offers two centralised pieces of infrastructure so individual
# routers can stop reinventing it:
#
# 1. `@handle_api_exceptions(...)` - a decorator that wraps a route handler
#    and converts common service-layer exceptions into `ApiException` with
#    consistent logging.
# 2. `unhandled_exception_handler` - a FastAPI exception handler intended to
#    be registered against `Exception` so anything that escapes a non-
#    decorated route still returns a structured JSON body instead of HTML.

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def handle_api_exceptions(
    *,
    action: Optional[str] = None,
    i18n_error_key: Optional[str] = None,
    value_error_code: int = 400,
    validation_error_code: int = 400,
    default_code: int = 500,
) -> Callable[[F], F]:
    """Convert service-layer exceptions into :class:`ApiException`.

    Args:
        action: Human-readable verb phrase woven into log lines and the
            default user-facing error message (e.g. ``"create knowledge
            base"``). Defaults to the wrapped function's name with
            underscores replaced by spaces.
        i18n_error_key: Optional i18n key (e.g. ``"api.llm.create_failed"``)
            used to render the user-facing message when an exception is
            converted. The original error is interpolated as the ``error``
            parameter: ``i18n.t(i18n_error_key, error=str(e))``. When
            ``None`` (the default), a plain English ``"Failed to {action}:
            {error}"`` message is used. Log lines always use the plain
            English form so they stay consistent across locales.
        value_error_code: HTTP status code used for :class:`ValueError`
            (the canonical "invalid input" signal in this codebase).
        validation_error_code: HTTP status code used for Pydantic
            :class:`pydantic.ValidationError` raised inside the handler
            (e.g. when constructing a response model from bad data).
        default_code: HTTP status code used for any other unexpected
            exception.

    Behaviour:
        - :class:`ApiException` / :class:`HTTPException` are re-raised as-is
          so endpoints can still pick a specific status code at the call
          site.
        - :class:`ValueError` is logged at WARNING and converted to
          ``ApiException(value_error_code, ...)``.
        - :class:`pydantic.ValidationError` is logged at WARNING and
          converted to ``ApiException(validation_error_code, ...)``.
        - All other exceptions are logged at ERROR with full traceback and
          converted to ``ApiException(default_code, ...)``.

    Example:
        >>> @router.get("")
        ... @handle_api_exceptions(action="get trace config")
        ... async def get_trace_config(...):
        ...     return success_response(data=await svc.get(...))

        >>> @router.post("")
        ... @handle_api_exceptions(
        ...     action="create llm", i18n_error_key="api.llm.create_failed"
        ... )
        ... async def create_llm(...):
        ...     return success_response(...)
    """

    def decorator(func: F) -> F:
        verb = action or func.__name__.replace("_", " ")

        def _user_message(err: Exception) -> str:
            if i18n_error_key:
                from common.i18n import i18n

                return i18n.t(i18n_error_key, error=str(err))
            return f"Failed to {verb}: {err}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Covers ApiException too (it subclasses HTTPException).
                # Routes that hand-craft a specific status code keep that
                # control instead of being collapsed to 500.
                raise
            except ValueError as e:
                logger.warning(f"Validation error while trying to {verb}: {e}")
                raise ApiException(
                    code=value_error_code, message=_user_message(e)
                ) from e
            except ValidationError as e:
                logger.warning(
                    f"Pydantic validation error while trying to {verb}: {e}"
                )
                raise ApiException(
                    code=validation_error_code, message=_user_message(e)
                ) from e
            except Exception as e:
                logger.error(
                    f"Failed to {verb}: {e}\n{traceback.format_exc()}"
                )
                raise ApiException(
                    code=default_code, message=_user_message(e)
                ) from e

        return wrapper  # type: ignore[return-value]

    return decorator


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Last-resort handler for exceptions that escape every other handler.

    Register against ``Exception`` in ``create_app`` so a route that forgets
    the decorator (or raises during dependency resolution) still gets a
    structured JSON body. Note: FastAPI dispatches to the most specific
    handler, so the existing :func:`api_exception_handler` and
    ``RequestValidationError`` handler continue to take precedence.
    """
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}: "
        f"{exc}\n{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": f"Internal server error: {type(exc).__name__}",
            "data": None,
        },
    )
