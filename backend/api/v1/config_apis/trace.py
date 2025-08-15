### Trace configuration API ###

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.trace import TraceModel, TraceModelEntity
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from extensions.trace.base import init_instrument, TraceConfig
from db.models.change_event import ChangeEventSource, ChangeEventType
from config.providers.config_change_manager import config_change_manager
from config.providers.trace_provider import trace_provider

from loguru import logger


trace_router = APIRouter()


@trace_router.post("", response_model=TraceModel)
async def set_trace_config(
    new_trace_config: TraceModel,
    session: AsyncSession = Depends(get_session),
):
    trace_config = (await session.exec(select(TraceModelEntity))).first()
    if trace_config is None:
        logger.info(f"Adding new trace config {trace_config}")

        trace_config = TraceModelEntity.model_validate(
            new_trace_config,
        )
    else:
        trace_config.endpoint = new_trace_config.endpoint or trace_config.endpoint
        trace_config.enabled = new_trace_config.enabled
        trace_config.token = new_trace_config.token or trace_config.token
        trace_config.service_name = (
            new_trace_config.service_name or trace_config.service_name
        )

    init_instrument(
        config=TraceConfig(
            service_name=trace_config.service_name,
            endpoint=trace_config.endpoint,
            token=trace_config.token,
            enabled=trace_config.enabled,
        )
    )

    session.add(trace_config)
    try:
        await session.commit()
        await session.refresh(trace_config)
        trace_provider.update(trace_config)
        config_change_manager.notify_change_async(
            event_source=ChangeEventSource.TRACE,
            source_id=trace_config.id,
            event_type=ChangeEventType.UPDATE,
        )
        return trace_config
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add search config: {e.orig}")
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=400, detail=f"Failed to add search config: {str(e)}"
        )


@trace_router.get("", response_model=TraceModel)
async def get_trace_config(
    session: AsyncSession = Depends(get_session),
):
    trace_config_results = await session.exec(select(TraceModelEntity))
    trace_config = trace_config_results.first()
    if not trace_config:
        logger.warning("No trace config found.")
        return TraceModel()

    return trace_config
