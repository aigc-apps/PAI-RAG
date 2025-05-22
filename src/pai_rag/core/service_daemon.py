import asyncio
import os
import subprocess
import traceback

from watchfiles import awatch
from pai_rag.core.chat_service import chat_service
from pai_rag.core.rag_environment import service_environment
from pai_rag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pai_rag.data_pipeline.job.rag_job_manager import job_manager, FileChange
from loguru import logger
from pai_rag.utils.constants import DEFAULT_KNOWLEDGEBASE_PATH
from pai_rag.web.filebrowser.constants import DEFAULT_FILE_BROWSER_PORT

# Check every 30 seconds.
CHECK_INTERVAL = 30
FILE_CHECK_INTERVAL = 10


async def periodic_check_config():
    logger.debug("Running periodic_check_config in background.")
    try:
        while True:
            knowledgebase_manager.check_updates()
            chat_service.check_updates()
            await asyncio.sleep(CHECK_INTERVAL)
    except Exception:
        logger.error(f"Error in periodic_check_config: {traceback.format_exc()}")
    finally:
        logger.info("Exited periodic check config updates.")


async def host_filebrowser_in_background():
    if service_environment.SHOULD_START_WEB:
        logger.debug("Starting up filebrowser in background.")
        process = subprocess.Popen(
            f"rm -rf /tmp/filebrowser.db && /bin/filebrowser -b /filebrowser --address 0.0.0.0 -p {DEFAULT_FILE_BROWSER_PORT} -r {DEFAULT_KNOWLEDGEBASE_PATH} --noauth -d /tmp/filebrowser.db",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # set non blocking read
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)

        while True:
            error = process.stderr.readline()
            while error:
                logger.warning(f"[filebrowser]: {error}")
                error = process.stderr.readline()

            line = process.stdout.readline()

            if not line:
                await asyncio.sleep(5)
                continue

            logger.info(f"[filebrowser]: {line}")

    else:
        logger.debug("Don't start up filebrowser in API mode.")

    logger.warning("filebrowser daemon exited.")


async def watch_knowledgebase_changes():
    async for changes in awatch(
        DEFAULT_KNOWLEDGEBASE_PATH,
        recursive=True,
        debounce=5000,
        force_polling=True,
        poll_delay_ms=2000,
    ):
        # change_type: 1 add, 2 modified, 3 delete.
        change_count = 0
        for change_type, file_path in changes:
            change_count += 1
            if change_count % 10 == 0:
                await asyncio.sleep(0.5)

            is_delete = change_type == 3
            try:
                knowledgebase, change_docs = knowledgebase_manager.get_change_files(
                    file_path, is_delete=is_delete
                )
            except Exception:
                logger.error(
                    f"Error when watching knowledgebase changes: {file_path}. Details:{traceback.format_exc()}"
                )
                continue
            if knowledgebase and len(change_docs) > 0:
                file_changes = [
                    FileChange(
                        task_id=doc.doc_id,
                        operation=change_type,
                        file_name=doc.file_name,
                        file_hash=doc.file_hash,
                        knowledgebase=knowledgebase,
                    )
                    for doc in change_docs
                ]
                job_manager.submit_job(file_changes)
                logger.info(
                    f"changes enqueued. {knowledgebase}, {file_changes}, {change_type}"
                )

        await asyncio.sleep(1)


async def startup_event():
    # 启动后台任务
    asyncio.create_task(host_filebrowser_in_background())
    asyncio.create_task(periodic_check_config())
    asyncio.create_task(watch_knowledgebase_changes())
