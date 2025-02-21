import os


from filelock import Timeout, FileLock
from loguru import logger

lock_file_path = "localdata/__shared_gradio_instance.lock"

# Start only one web instance


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
        except Exception as ex:
            logger.info(f"Unhandled error when acquiring lock: {ex}")
            return False

    def release(self):
        if self.has_lock:
            try:
                self.lock.release()
                logger.info("Release lock successfully.")
            except Exception as ex:
                logger.info(f"Release lock failed: {ex}")
                raise ex


class RagServiceEnvironment:
    def __init__(self):
        self.IS_API_INSTANCE = os.getenv("DEPLOY_MODE", "web").upper() == "API"
        self.SHOULD_START_WEB = False
        self._lock = None
        if not self.IS_API_INSTANCE:
            self._lock = WebInstanceLock()
            if self._lock.try_acquire():
                logger.info("Acuiqred lock successfully.")
                self.SHOULD_START_WEB = True

    def cleanup(self):
        self._lock.release()


service_environment = RagServiceEnvironment()
