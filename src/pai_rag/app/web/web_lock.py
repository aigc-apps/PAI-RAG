from filelock import Timeout, FileLock
from loguru import logger

lock_file_path = "localdata/__shared_gradio_instance.lock"


class WebInstanceLock:
    def __init__(self):
        open(lock_file_path, "w").write("abc")
        self.lock = FileLock(lock_file_path, timeout=2)
        self.has_lock = False

    def try_acquire(self):
        try:
            self.lock.acquire()
            self.has_lock = True
            return True
        except Timeout:
            logger.info("Acuiqre lock failed due to timeout.")
            return False

    def try_release(self):
        if self.has_lock:
            try:
                self.lock.release()
                return True
            except Exception as ex:
                logger.info(f"Release lock failed: {ex}")
                return False
