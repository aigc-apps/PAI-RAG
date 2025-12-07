from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Any, Optional
from pydantic import BaseModel
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
