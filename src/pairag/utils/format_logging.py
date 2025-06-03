import logging
import sys
from loguru import logger
from asgi_correlation_id.context import correlation_id


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        # 尝试获取与标准 logging 等级相对应的 Loguru 日志等级
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            # 如果找不到对应的 Loguru 等级，则使用原始的数字等级
            level = record.levelno

        # 探测调用日志的代码位置
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        # 使用 Loguru 记录日志信息，保持调用栈的深度和异常信息
        logger.opt(
            depth=depth,
            exception=record.exc_info,
        ).log(level, record.getMessage())


# 自定义日志格式，加入 request_id
def formatter(record):
    record["extra"]["request_id"] = correlation_id.get()
    if record["extra"].get("request_id") is None:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{process}</level> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>\n"
        )
    else:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{process}</level> | "
            "<level>{extra[request_id]} |</level> "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>\n"
        )


def format_logging():
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    logger.remove(0)
    logger.add(
        sys.stderr,
        format=formatter,
    )
