### Chat bi configuration API ###

import traceback
from typing import List
from db.models.chatdb.chatdb import ChatDbConfigEntity, ChatDbCreate
from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from common.chat.response_model import ResponseModel, success_response
from db.db_context import get_db_session
from common.encrypt_utils import decrypt_key
from service.tool.chatdb_service import ChatdbService
from service.injection import get_chatdb_service
from api.api_exception import ApiException
from urllib.parse import quote_plus
from loguru import logger


chatdb_router = APIRouter()


@chatdb_router.post("", response_model=ResponseModel[ChatDbConfigEntity])
async def add_chatdb_config(
    new_db_config: ChatDbCreate,
    session: AsyncSession = Depends(get_db_session),
    chatdb_service: ChatdbService = Depends(get_chatdb_service),
):
    if new_db_config.dialect.lower() not in ["mysql", "postgresql"]:
        logger.error(f"不支持的数据库类型{new_db_config.dialect}，仅支持mysql和postgresql")
        raise ApiException(code=400, message=f"不支持的数据库类型{new_db_config.dialect}，仅支持mysql和postgresql")

    try:
        chatdb_config = await chatdb_service.create_or_update_chatdb_config(new_db_config)
        await session.refresh(chatdb_config)
        return success_response(data=chatdb_config, message="添加ChatDB配置成功!")
    except ValueError as e:
        logger.error(f"Failed to add chatdb config: {str(e)}")
        raise ApiException(code=400, message=f"添加ChatDB配置失败: {e}")
    except Exception as e:
        logger.error(f"Failed to add chatdb config: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"添加ChatDB配置失败: {e}")

@chatdb_router.get("", response_model=ResponseModel[List[ChatDbConfigEntity]])
async def list_chatdb_config(
    session: AsyncSession = Depends(get_db_session),
    chatdb_service: ChatdbService = Depends(get_chatdb_service),
):
    try:
        configs = await chatdb_service.get_all_chatdb_configs()
        return success_response(data=configs, message="获取ChatDB配置成功!")
    except Exception as e:
        logger.error(f"获取ChatDB配置失败: {traceback.format_exc()}。")
        raise ApiException(code=400, message=f"获取ChatDB配置失败: {e}。")


@chatdb_router.post("/connectiontest")
async def connection_test(
    db_config: ChatDbCreate,
    session: AsyncSession = Depends(get_db_session),
    chatdb_service: ChatdbService = Depends(get_chatdb_service),
):
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError

    if not db_config.password:
        existing_config = await chatdb_service.get_chatdb_config_or_create()
        if existing_config:
            db_config.password = decrypt_key(existing_config.encrypted_password)
    db_config.dialect = db_config.dialect.lower()

    if db_config.dialect == "mysql":
        db_url = f"mysql+pymysql://{db_config.username}:{quote_plus(db_config.password)}@{db_config.host}:{db_config.port}/{db_config.db_name}"
    elif db_config.dialect == "postgresql":
        db_url = f"postgresql+psycopg2://{db_config.username}:{quote_plus(db_config.password)}@{db_config.host}:{db_config.port}/{db_config.db_name}"
    else:
        raise ApiException(code=400, message=f"不支持的数据库类型{db_config.dialect}, 仅支持mysql和postgresql")

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
        raise ApiException(code=400, message=f"数据库连接失败: {e}")
    except Exception as e:
        logger.error(f"未知错误: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"数据库连接失败: {e}")
    finally:
        if engine:
            engine.dispose()
            logger.info("数据库连接已关闭")
