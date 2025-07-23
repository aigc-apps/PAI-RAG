import logging
import sys
import uuid
from opentelemetry import trace
from loguru import logger

from pairag.chat.chat_context import get_context


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


def get_current_trace_args() -> dict:
    user_args = {}
    current_span = trace.get_current_span()

    if current_span is not None:
        trace_number = current_span.get_span_context().trace_id
        if trace_number > 0:
            user_args["trace_id"] = uuid.UUID(int=trace_number).hex

    context_args = get_context()
    user_args.update(**context_args)
    return user_args


# 自定义日志格式，加入 request_id
def formatter(record):
    formatter_part_1 = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{process}</level> | "
    )
    formatter_part_2 = "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"
    user_args = get_current_trace_args()
    trace_template_str = ""
    for k, v in user_args.items():
        if v:
            record["extra"][k] = f"{k} {v}"
            trace_template_str += f"<cyan>{{extra[{k}]}}</cyan> | "
    return formatter_part_1 + trace_template_str + formatter_part_2


def format_logging():
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    logger.remove(0)
    logger.add(
        sys.stderr,
        format=formatter,
    )
