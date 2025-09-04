import os
import shutil
import threading
import time
import traceback

from loguru import logger


MOUNT_PATH = "./localdata"
RUNTIME_PATH = "./tmp/sqlite"
LOCAL_DB_FILE = "local.db"
LOCAL_CHROMA_FOLDER = "chroma"

os.makedirs(RUNTIME_PATH, exist_ok=True)
mount_db_file = os.path.join(MOUNT_PATH, LOCAL_DB_FILE)
mount_chroma_dir = os.path.join(MOUNT_PATH, LOCAL_CHROMA_FOLDER)

runtime_db_file = os.path.join(RUNTIME_PATH, LOCAL_DB_FILE)
runtime_chroma_dir = os.path.join(RUNTIME_PATH, LOCAL_CHROMA_FOLDER)


stop_event = threading.Event()


def init_sqlite_store():
    if os.path.exists(mount_db_file):
        shutil.copy2(mount_db_file, runtime_db_file)
    if os.path.exists(mount_chroma_dir):
        shutil.copytree(mount_chroma_dir, runtime_chroma_dir ,dirs_exist_ok=True)
    logger.info("Successfully copied sqlite store from mount path.")


def sync_sqlite_store():
    if os.path.exists(runtime_db_file):
        shutil.copy2(runtime_db_file, mount_db_file)
    if os.path.exists(runtime_chroma_dir):
        shutil.copytree(runtime_chroma_dir, mount_chroma_dir, dirs_exist_ok=True)


def sync_sqlite_store_task():
    count = 0
    logger.info("Start sqlite sync task.")

    while not stop_event.is_set():
        count += 1
        if count % 360 == 0:
            logger.info("Start syncing sqlite data in background")

        if count % 12 == 0:
            # try to sync every minutes.
            try:
                sync_sqlite_store()
            except Exception:
                logger.error(f"Sync sqlite store failed: {traceback.format_exc()}.")

        if count % 360 == 0:
            logger.info("Finished syncing sqlite data in background")
            count = 0

        time.sleep(5)

    logger.info("Stop sqlite sync task.")
