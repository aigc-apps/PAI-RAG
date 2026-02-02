from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from loguru import logger
from common.i18n import i18n


def format_validation_errors(errors: List[Dict[str, Any]]) -> str:
    """
    格式化验证错误为可读字符串

    示例输出：
    "name: 字段长度必须大于等于 3 个字符; description: 字段不能为空"
    """
    error_messages = []

    for error in errors:
        # 获取字段路径
        loc = error.get("loc", [])
        field = " -> ".join(str(x) for x in loc[1:]) if len(loc) > 1 else str(loc[0])

        # 获取错误消息
        msg = error.get("msg", "")
        error_type = error.get("type", "")

        # 翻译常见的错误类型
        friendly_msg = translate_error_message(msg, error_type, error)

        error_messages.append(f"{field}: {friendly_msg}")

    return "; ".join(error_messages)

def translate_error_message(msg: str, error_type: str, error: Dict) -> str:
    """
    翻译 Pydantic 错误消息为中文
    """
    translations = {
        "value_error.missing": "字段不能为空",
        "type_error.none.not_allowed": "字段不能为 null",
        "value_error.str.regex": "字段格式不正确",
        "value_error.any_str.min_length": "字段长度不能少于 {limit_value} 个字符",
        "value_error.any_str.max_length": "字段长度不能超过 {limit_value} 个字符",
        "value_error.number.not_gt": "数值必须大于 {limit_value}",
        "value_error.number.not_ge": "数值必须大于等于 {limit_value}",
        "type_error.integer": "字段必须是整数",
        "type_error.float": "字段必须是数字",
    }

    # 尝试翻译
    if error_type in translations:
        template = translations[error_type]
        # 替换模板变量
        ctx = error.get("ctx", {})
        try:
            return template.format(**ctx)
        except Exception:
            return template

    # 返回原始消息
    return msg

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    自定义 422 验证错误处理器
    打印详细的错误信息
    """
    # 获取请求详情
    body = None
    try:
        body = await request.body()
    except Exception as e:
        # 记录详细日志
        logger.warning(f"Request Body Parse Error: {e}")
        pass

    # 记录详细日志
    logger.warning(
        f"Validation Error:\n"
        f"  Path: {request.url.path}\n"
        f"  Method: {request.method}\n"
        f"  Client: {request.client.host if request.client else 'Unknown'}\n"
        f"  Headers: {dict(request.headers)}\n"
        f"  Query Params: {dict(request.query_params)}\n"
        f"  Body: {body.decode() if body else 'None'}\n"
        f"  Errors: {exc.errors()}"
    )


    exc_errors = exc.errors()
    if len(exc_errors) > 0:
        if isinstance(exc_errors[0], ValueError):
            error_message = str(exc_errors[0])
        elif isinstance(exc_errors[0], dict) and exc_errors[0].get("type") == "value_error":
            error_message = exc_errors[0].get("msg")
        else:
            error_message = format_validation_errors(exc_errors)
    else:
        error_message = i18n.t("api.error.validation")

    # 返回格式化的错误响应
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "code": 400,
            "message": error_message,
            "data": None
        }
    )
