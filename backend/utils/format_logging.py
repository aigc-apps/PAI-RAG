import logging
import sys
from extensions.trace.context import get_request_id
from loguru import logger

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # 获取对应的 Loguru 等级
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 优化：动态探测堆栈深度，彻底解决定位到 logging 内部的问题
        # 寻找第一个不在 logging 模块内的调用帧
        frame = logging.currentframe()
        depth = 0
        while frame:
            # 这里的逻辑是：只要文件名里包含 "logging"，就继续往上找
            if "logging" in frame.f_code.co_filename:
                frame = frame.f_back
                depth += 1
            else:
                break

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def generic_formatter(record, tag="API"):
    """
    通用格式化器：合并了 API 和 WORKER 的逻辑
    """
    rid = get_request_id()
    record["extra"]["request_id"] = rid

    # 基础模板
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{process}</level> | "

    # 如果有 request_id 则添加
    if rid:
        fmt += "<level>{extra[request_id]}</level> | "

    # 添加等级、标签和位置信息
    fmt += f"<level>{{level: <8}}</level> | <level>{tag}</level> | "
    fmt += "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"

    return fmt

def format_logging():
    # force=True 确保清除之前的配置，防止重复打印
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    logger.remove()
    logger.add(
        sys.stderr,
        format=lambda r: generic_formatter(r, tag="API"),
    )

def format_worker_logging():
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    logger.remove() # 使用不带参数的 remove 清除所有 handler
    logger.add(
        sys.stderr,
        format=lambda r: generic_formatter(r, tag="WORKER"),
    )
