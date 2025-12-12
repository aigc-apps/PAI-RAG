import os
from pairag.file.store.base import BaseFileStore
from pairag.file.store.local_store import LocalFileStore
from pairag.file.store.oss_store import DEFAULT_OSS_PREFIX, OssFileStore
from pairag.file.utils.constants import DEFAULT_KNOWLEDGEBASE_PATH
from pairag.file.store.bailian_file_store import BailianFileStore
from loguru import logger


def create_file_store_from_env() -> BaseFileStore:
    """
    根据环境变量创建文件存储实例。

    :return: 文件存储实例
    """
    logger.info("Creating file store from environment variables...")
    file_store_type = os.getenv("FILE_STORE_TYPE", "local").lower()
    if file_store_type == "oss":
        bucket = os.getenv("OSS_BUCKET")
        endpoint = os.getenv("OSS_ENDPOINT")
        prefix_path = os.getenv("OSS_PREFIX_PATH", DEFAULT_OSS_PREFIX)

        assert bucket, "OSS_BUCKET environment variable is required."
        assert endpoint, "OSS_ENDPOINT environment variable is required."

        logger.info("Created OSS file store.")
        return OssFileStore(bucket=bucket, endpoint=endpoint, prefix_path=prefix_path)
    elif file_store_type == "bailian":
        return BailianFileStore()
    else:
        if file_store_type != "local":
            logger.warning(
                f"Unknown FILE_STORE_TYPE {file_store_type}, using local file store."
            )
        logger.info("Created local file store.")
        return LocalFileStore(base_path=DEFAULT_KNOWLEDGEBASE_PATH)


file_store = create_file_store_from_env()
