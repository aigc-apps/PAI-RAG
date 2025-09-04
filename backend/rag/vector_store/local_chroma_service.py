from loguru import logger
import subprocess
import requests
from utils.constants import try_get_int_env


DEFAULT_CHROMA_PORT = try_get_int_env("CHROMA_PORT", 8684)


class LocalChromaService:
    def __init__(self, data_path: str = "./tmp/sqlite/chroma", port=8684):
        self.data_path = data_path
        self.port = port
        self.proc = None

    def verify_chroma_is_running(self):
        try:
            response = requests.get(f"http://localhost:{self.port}/docs")
            assert response.status_code == 200
            logger.warning("Chroma is already running.")
            return True
        except Exception:
            logger.warning("Chroma is not running.")
            return False

    def start(self):
        if not self.verify_chroma_is_running():
            logger.warning("Starting Chroma...")
            self.proc = subprocess.Popen([
                'chroma',
                'run',
                '--path',
                self.data_path,
                '--port',
                str(self.port),
            ])

            logger.info(f"Chroma started at port {self.port}. 子进程 PID: {self.proc.pid}")

    def stop(self):
        if self.proc:
            self.proc.terminate()  # 或 proc.kill() 强制杀死
            self.proc.wait()       # 等待真正退出
        logger.info("Chroma stopped.")
