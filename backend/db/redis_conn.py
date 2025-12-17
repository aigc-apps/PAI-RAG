import os
from urllib.parse import quote_plus, urlunparse
from typing import Optional
from loguru import logger

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
if not REDIS_HOST:
    REDIS_HOST = "localhost"
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
if not REDIS_PORT:
    REDIS_PORT = 6379
else:
    REDIS_PORT = int(REDIS_PORT)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_USER=os.getenv("REDIS_USER", "")
REDIS_DB = os.getenv("REDIS_DB")

if not REDIS_DB:
    REDIS_DB = 0
else:
    REDIS_DB = int(REDIS_DB)


def compose_redis_url_safe(
    host: str = "localhost",
    port: int = 6379,
    username: Optional[str] = None,
    password: Optional[str] = None,
    db: int = 0,
    ssl: bool = False
) -> str:
    """
    安全地组合 Redis URL（处理特殊字符）

    这个版本会自动对用户名和密码中的特殊字符进行URL编码
    """
    # 协议
    scheme = "rediss" if ssl else "redis"

    # 处理认证信息（URL编码特殊字符）
    if username and password:
        netloc = f"{quote_plus(username)}:{quote_plus(password)}@{host}:{port}"
    elif password:
        netloc = f":{quote_plus(password)}@{host}:{port}"
    else:
        netloc = f"{host}:{port}"

    # 路径（数据库编号）
    path = f"/{db}"

    # 组合完整URL
    url = urlunparse((scheme, netloc, path, '', '', ''))

    return url


REDIS_URL = compose_redis_url_safe(
    host=REDIS_HOST,
    port=REDIS_PORT,
    username=REDIS_USER,
    password=REDIS_PASSWORD,
    db=REDIS_DB,
    ssl=False,
)
logger.info(f"Redis URL: {REDIS_URL}")
