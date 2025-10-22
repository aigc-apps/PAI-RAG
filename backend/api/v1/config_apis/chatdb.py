### Chat bi configuration API ###

import traceback
from typing import List
from db.models.chatdb.chatdb import ChatDbConfigEntity, ChatDbCreate
from fastapi import APIRouter, Depends, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from api.response_model import ResponseModel, error_response, success_response
from db.db_context import get_session
from common.encrypt_utils import decrypt_key, encrypt_key
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from config.providers.chatdb_provider import chatdb_provider
from urllib.parse import quote_plus
from loguru import logger


chatdb_router = APIRouter()


@chatdb_router.post("", response_model=ResponseModel[ChatDbConfigEntity])
async def add_chatdb_config(
    new_db_config: ChatDbCreate,
    session: AsyncSession = Depends(get_session),
):
    new_db_config.dialect = new_db_config.dialect.lower()

    if new_db_config.dialect not in ["mysql", "postgresql"]:
        return error_response(code=400, message="不支持的数据库类型，仅支持mysql和postgresql")

    encrypted_password = encrypt_key(new_db_config.password)

    statement = select(ChatDbConfigEntity)
    chatdb_config = (await session.exec(statement)).first()
    if chatdb_config is None:
        logger.info(f"Adding new chatdb config for dialect {new_db_config.dialect}")

        chatdb_config = ChatDbConfigEntity.model_validate(
            new_db_config,
            update={
                "encrypted_password": encrypted_password,
            },
        )
    else:
        chatdb_config.dialect = new_db_config.dialect
        chatdb_config.db_name = new_db_config.db_name
        chatdb_config.username = new_db_config.username
        chatdb_config.encrypted_password = encrypted_password or chatdb_config.encrypted_password
        chatdb_config.model_id = new_db_config.model_id
        chatdb_config.host = new_db_config.host
        chatdb_config.port = new_db_config.port

    session.add(chatdb_config)
    try:
        await session.commit()
        await session.refresh(chatdb_config)
        chatdb_provider.update(chatdb_config)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.CHATDB,
            source_id=chatdb_config.id,
            event_type=ChangeEventType.UPDATE,
        )
        return success_response(data=chatdb_config, message="添加ChatDB配置成功!")

    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add chatdb config: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"Failed to add chatdb config: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to add chatdb config: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"Failed to add chatdb config: {str(e)}")

@chatdb_router.get("", response_model=ResponseModel[List[ChatDbConfigEntity]])
async def list_chatdb_config(
    session: AsyncSession = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    try:
        chatdb_config = (await session.exec(
            select(ChatDbConfigEntity).offset(offset).limit(limit)
        )).first()
        if chatdb_config:
            result = [chatdb_config]
        else:
            result = []

        return success_response(data=result, message="获取ChatDB配置成功!")
    except Exception as e:
        logger.error(f"获取ChatDB配置失败: {traceback.format_exc()}。")
        return error_response(code=400, message=f"获取ChatDB配置失败: {e}。")


@chatdb_router.post("/connectiontest")
async def connection_test(
    db_config: ChatDbCreate,
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError

    if not db_config.password:
        existing_config = (await session.exec(
                select(ChatDbConfigEntity)
            )).first()
        if existing_config:
            db_config.password = decrypt_key(existing_config.encrypted_password)
    db_config.dialect = db_config.dialect.lower()

    if db_config.dialect == "mysql":
        db_url = f"mysql+pymysql://{db_config.username}:{quote_plus(db_config.password)}@{db_config.host}:{db_config.port}/{db_config.db_name}"
    elif db_config.dialect == "postgresql":
        db_url = f"postgresql+psycopg2://{db_config.username}:{quote_plus(db_config.password)}@{db_config.host}:{db_config.port}/{db_config.db_name}"
    else:
        return error_response(code=400, message=f"不支持的数据库类型{db_config.dialect}, 仅支持mysql和postgresql")

    try:
        # 添加连接参数（设置超时）
        connect_args = {
            "connect_timeout": 5,
            "charset": "utf8mb4"
        }

        engine = create_engine(
            db_url,
            connect_args=connect_args,
            pool_pre_ping=True,  # 每次从池获取连接时先 ping 一下
            pool_recycle=3600    # 1 小时回收连接
        )

        # 尝试获取连接并执行简单查询
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return success_response(message="连接成功")
    except SQLAlchemyError as e:
        logger.error(f"数据库连接失败: {traceback.format_exc()}")
        return error_response(code=400, message=f"数据库连接失败: {e}")
    except Exception as e:
        logger.error(f"未知错误: {traceback.format_exc()}")
        return error_response(code=500, message=f"数据库连接失败: {e}")
