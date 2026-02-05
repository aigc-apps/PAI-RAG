from datetime import datetime


def get_current_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def get_current_time_str_zh() -> str:
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S %A")
