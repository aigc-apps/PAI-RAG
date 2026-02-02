from typing import Generic, TypeVar, Optional
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from common.i18n import i18n


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
    else:
        return obj


def success_response(code=200, data=None, message=None):
    if message is None:
        message = i18n.t("api.success.operation")
    content = {"code": code, "message": message, "data": to_dict(data)}
    return JSONResponse(status_code=code, content=jsonable_encoder(content))
