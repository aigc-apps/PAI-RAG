### Trace configuration API ###

from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.trace import TraceModel
from db.db_context import get_db_session
from service.tool.trace_service import TraceService
from service.injection import get_trace_service, get_tenant_id
from common.chat.response_model import success_response
from api.api_exception import handle_api_exceptions
from loguru import logger


trace_router = APIRouter()


@trace_router.post("", response_model=TraceModel)
@handle_api_exceptions(action="create or update trace config")
async def set_trace_config(
    new_trace_config: TraceModel,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    trace_service: TraceService = Depends(get_trace_service),
):
    trace_config = await trace_service.create_or_update_trace_config(
        new_trace_config=new_trace_config, tenant_id=tenant_id
    )
    return success_response(data=trace_config, message="Update trace config success.")


@trace_router.get("", response_model=TraceModel)
@handle_api_exceptions(action="get trace config")
async def get_trace_config(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    trace_service: TraceService = Depends(get_trace_service),
):
    trace_config = await trace_service.get_trace_config(tenant_id=tenant_id)
    if not trace_config:
        logger.warning("No trace config found.")
        return success_response(data=TraceModel(), message="Get trace config success.")
    return success_response(data=trace_config, message="Get trace config success.")
