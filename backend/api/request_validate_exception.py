from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from fastapi import Request
from loguru import logger
from fastapi import status

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
        logger.warning(f"Failed to parse request body: {e}")
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

    # 返回格式化的错误响应
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": jsonable_encoder(exc.errors()),
            "body": body.decode() if body else None
        }
    )
