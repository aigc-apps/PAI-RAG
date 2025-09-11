from datetime import datetime
import json
import os
import shutil
import threading
import time
import traceback

from loguru import logger


OLD_MOUNT_PATH = "./localdata"
MOUNT_PATH = "./localdata/sqlite"
RUNTIME_PATH = "./tmp/sqlite"
LOCAL_DB_FILE = "local.db"
LOCAL_CHROMA_FOLDER = "chroma"

SQLITE_CACHE_FILE = os.path.join(MOUNT_PATH, "cache_version.json")
SQLITE_CACHE_PATH = os.path.join(MOUNT_PATH, "cache")


os.makedirs(RUNTIME_PATH, exist_ok=True)
old_mount_db_file = os.path.join(OLD_MOUNT_PATH, LOCAL_DB_FILE)
old_mount_chroma_dir = os.path.join(OLD_MOUNT_PATH, LOCAL_CHROMA_FOLDER)

runtime_db_file = os.path.join(RUNTIME_PATH, LOCAL_DB_FILE)
runtime_chroma_dir = os.path.join(RUNTIME_PATH, LOCAL_CHROMA_FOLDER)


stop_event = threading.Event()


def init_sqlite_store():
    if not os.path.exists(SQLITE_CACHE_FILE):
        if os.path.exists(old_mount_db_file) and not os.path.exists(runtime_db_file):
            shutil.copy2(old_mount_db_file, runtime_db_file)
            logger.info(f"Copied data from {old_mount_db_file} to {runtime_db_file}")
        if os.path.exists(old_mount_chroma_dir) and not os.path.exists(runtime_chroma_dir):
            shutil.copytree(old_mount_chroma_dir, runtime_chroma_dir, dirs_exist_ok=True)
            logger.info(f"Copied data from {old_mount_chroma_dir} to {runtime_chroma_dir}")
    elif not os.path.exists(runtime_db_file):
        try:
            cache_version = None
            with open(SQLITE_CACHE_FILE, "r") as rf:
                cache_obj = json.loads(rf.read())
                cache_version = cache_obj.get("version", None)

            if cache_version:
                local_cache_dir = os.path.join(SQLITE_CACHE_PATH, cache_version)
                if os.path.exists(local_cache_dir):
                    shutil.copytree(local_cache_dir, RUNTIME_PATH, dirs_exist_ok=True)
                    logger.info(f"Copied data from {local_cache_dir} to {RUNTIME_PATH}")
        except Exception:
            logger.error(f"同步sqlite缓存出错: {traceback.format_exc()}")

    logger.info("Successfully inited sqlite store.")



def sync_sqlite_store():
    logger.info("Starting persist sqlite data.")
    current_date_key = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    local_cache_dir = os.path.join(SQLITE_CACHE_PATH, current_date_key)
    os.makedirs(local_cache_dir, exist_ok=True)
    shutil.copytree(RUNTIME_PATH, local_cache_dir, dirs_exist_ok=True)

    cache_version_text = json.dumps({"version": current_date_key})
    with open(SQLITE_CACHE_FILE, 'w') as wf:
        wf.write(cache_version_text)

    logger.info("Persist sqlite data success.")
    for dir_name in os.listdir(SQLITE_CACHE_PATH):
        if dir_name < current_date_key:
            cache_dir_to_remove = os.path.join(SQLITE_CACHE_PATH, dir_name)
            shutil.rmtree(cache_dir_to_remove)
            logger.info(f"Removed cache dir {cache_dir_to_remove}.")



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
