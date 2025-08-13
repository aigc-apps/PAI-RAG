from typing import Generic, TypeVar, Optional
from fastapi.responses import JSONResponse
from pydantic import BaseModel


T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):
    code: int
    message: str
    data: Optional[T] = None

class PagedResult(BaseModel, Generic[T]):
    items: Optional[T] = None
    total: int
    pages: int
    page: int
    size: int

def to_dict(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(v) for v in obj]


def success_response(code=200, data=None, message="操作成功"):
    return {"code": code, "message": message, "data": to_dict(data)}


def error_response(code=500, data=None, message="系统错误"):
    return JSONResponse(status_code=code, content={"code": code, "message": message, "data": to_dict(data)})
