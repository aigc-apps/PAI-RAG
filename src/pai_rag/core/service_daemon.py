import asyncio
from pai_rag.core.rag_service import rag_service
from pai_rag.core.rag_index_manager import index_manager
from loguru import logger
import traceback

# Check every 30 seconds.
CHECK_INTERVAL = 30


async def periodic_check_config():
    logger.debug("Running periodic_usage_pool_cleanup")
    try:
        while True:
            index_manager.check_updates()
            rag_service.check_updates()
            await asyncio.sleep(CHECK_INTERVAL)
    except Exception:
        logger.error(f"Error in periodic_check_config: {traceback.format_exc()}")
    finally:
        logger.info("Exited periodic check config updates.")
