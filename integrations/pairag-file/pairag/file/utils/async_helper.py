import asyncio
import concurrent.futures
import threading
from loguru import logger

def run_sync(coro):
    """
    在同步上下文中运行异步协程。
    如果事件循环已经在运行，则在新的线程中运行。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 无运行中 loop → 直接 run
        return asyncio.run(coro)
    else:
        # 有运行中 loop（如 uvloop + FastAPI/Jupyter）→ 新线程运行
        def _run():
            # 新线程无 loop，可安全 asyncio.run
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, 
            thread_name_prefix="run_sync"
        ) as executor:
            return executor.submit(_run).result()