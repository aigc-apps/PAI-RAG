from datetime import datetime
from typing import Optional


def get_current_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_prompt_current_time_str() -> str:
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


def get_timestamp(dt: Optional[datetime] = None) -> int:
    if dt is None:
        return int(datetime.now().timestamp() * 1000) # 毫秒级时间戳
    return int(dt.timestamp() * 1000) # 毫秒级时间戳
