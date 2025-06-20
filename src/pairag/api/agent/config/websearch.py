### Web search configuration API ###

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models.websearch import (
    WebSearchConfigRead,
    WebSearchConfigCreate,
    WebSearchConfigEntity,
)
from pairag.db.db_context import get_session
from pairag.db.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError

from loguru import logger


websearch_router = APIRouter()


@websearch_router.post("", response_model=WebSearchConfigRead)
async def add_search_config(
    new_search_config: WebSearchConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    encrypted_access_key_id = encrypt_key(new_search_config.access_key_id)
    encrypted_access_key_secret = encrypt_key(new_search_config.access_key_secret)

    statement = select(WebSearchConfigEntity).where(
        WebSearchConfigEntity.type == new_search_config.type
    )
    search_config = (await session.exec(statement)).first()
    if search_config is None:
        logger.info(f"Adding new search config for type {new_search_config.type}")

        search_config = WebSearchConfigEntity.model_validate(
            new_search_config,
            update={
                "encrypted_access_key_id": encrypted_access_key_id,
                "encrypted_access_key_secret": encrypted_access_key_secret,
            },
        )
    else:
        search_config.encrypted_access_key_id = encrypted_access_key_id
        search_config.encrypted_access_key_secret = encrypted_access_key_secret
        search_config.endpoint = new_search_config.endpoint or search_config.endpoint

    session.add(search_config)
    try:
        await session.commit()
        await session.refresh(search_config)
        return search_config
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add search config: {e.orig}")
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=400, detail=f"Failed to add search config: {str(e)}"
        )


@websearch_router.get("", response_model=List[WebSearchConfigRead])
async def list_search_config(
    session: AsyncSession = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    search_config_results = await session.exec(
        select(WebSearchConfigEntity).offset(offset).limit(limit)
    )
    return search_config_results.all()
