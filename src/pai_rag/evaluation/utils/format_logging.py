import logging
from loguru import logger


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
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def format_logging():
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
