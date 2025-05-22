import os
import shutil
from loguru import logger


def clear_folder(folder_path: str):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        logger.warning(
            f"Fail to clear path {folder_path} because it is not a directory or does not exist."
        )
        return

    shutil.rmtree(folder_path)  # 删除整个文件夹
    os.makedirs(folder_path)  # 创建新的空文件夹
    logger.info(f"Folder {folder_path} cleared successfully.")

    return
