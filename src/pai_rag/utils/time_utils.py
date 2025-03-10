from datetime import datetime


def get_current_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
