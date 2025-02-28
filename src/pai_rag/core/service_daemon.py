import asyncio
import traceback
from pai_rag.core.rag_service import rag_service
from pai_rag.core.rag_index_manager import index_manager
from pai_rag.app.web.rag_local_client import rag_client
from pai_rag.core.rag_index_manager import batch_files, batch_lock
from loguru import logger

# Check every 30 seconds.
CHECK_INTERVAL = 30
FILE_CHECK_INTERVAL = 10


async def periodic_check_config():
    logger.debug("Running periodic_check_config")
    try:
        while True:
            index_manager.check_updates()
            rag_service.check_updates()
            await asyncio.sleep(CHECK_INTERVAL)
    except Exception:
        logger.error(f"Error in periodic_check_config: {traceback.format_exc()}")
    finally:
        logger.info("Exited periodic check config updates.")


# 定义后台任务
async def process_batch_files():
    while True:
        await asyncio.sleep(FILE_CHECK_INTERVAL)  # 等待15秒
        with batch_lock:
            if not batch_files:
                continue  # 如果没有文件，则跳过
            current_batch = batch_files.copy()
            batch_files.clear()
        logger.info(f"Processing current_batch: {current_batch}")
        try:
            tasks = [
                rag_client.async_add_knowledge_file(
                    index_name=index_name, input_files=file_paths
                )
                for index_name, file_paths in current_batch.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")


async def startup_event():
    # 启动后台任务
    logger.debug("Running process_batch_files")
    asyncio.create_task(process_batch_files())
