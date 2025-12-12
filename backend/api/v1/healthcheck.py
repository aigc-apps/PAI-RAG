import time
from typing import Any, Dict
from fastapi import APIRouter
from common.chat.response_model import ResponseModel, success_response
from pydantic import BaseModel


health_router = APIRouter()


class HealthCheck(BaseModel):
    status: str
    timestamp: float
    service: str
    checks: Dict[str, Any] = {}


# 全局变量记录启动时间
start_time = time.time()

@health_router.get("", response_model=ResponseModel[HealthCheck], tags=["Health"])
async def health_check():
    """基础健康检查"""
    return success_response(
        data=HealthCheck(
            status="healthy",
            timestamp=time.time(),
            service="PAI-RAG"
        ),
        message="Health check success!")
