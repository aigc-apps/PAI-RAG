from io import BytesIO
import os
from typing import BinaryIO, Optional
from pairag.store.file.base import BaseFileStore
from loguru import logger


class LocalFileStore(BaseFileStore):
    def __init__(self, base_path: str):
        super().__init__()

        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        elif not os.path.isdir(base_path):
            raise ValueError("base_path must be a directory")

        self.base_path = base_path

    def get_url(self, file_path: str):
        return os.path.join(self.base_path, file_path)

    def save(self, data: BinaryIO, file_path: str) -> None:
        full_path = os.path.join(self.base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(data.read())
        logger.info(f"Saved file to {full_path}.")

    def load(self, file_path: str) -> Optional[BinaryIO]:
        full_path = os.path.join(self.base_path, file_path)

        with open(full_path, "rb") as f:
            return BytesIO(f.read())

    def exists(self, file_path: str) -> bool:
        full_path = os.path.join(self.base_path, file_path)
        return os.path.exists(full_path)
