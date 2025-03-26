import os
import time
import fcntl
from loguru import logger
from pai_rag.utils.download_models import ModelScopeDownloader


def download_models_via_lock(model_dir, model_name, accelerator="cpu"):
    model_path = os.path.join(model_dir, model_name)
    lock_file_path = model_name + ".lock"
    # 创建或打开一个锁文件
    with open(lock_file_path, "w") as lock_file:
        while True:
            try:
                # 尝试获取文件锁
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info(f"进程 {os.getpid()} 获得锁")

                # 检查模型文件是否已经下载
                if os.path.exists(model_path):
                    logger.info(f"进程 {os.getpid()} 检查到: 模型已下载完成，环境: {accelerator}。")
                else:
                    logger.info(f"进程 {os.getpid()} 开始下载模型，环境: {accelerator}。")
                    ModelScopeDownloader(
                        fetch_config=True,
                        download_directory_path=model_dir,
                    ).load_model(model=model_name)
                    if model_name == "PDF-Extract-Kit":
                        ModelScopeDownloader(
                            fetch_config=True,
                            download_directory_path=model_dir,
                        ).load_mineru_config(accelerator)
                    logger.info(f"进程 {os.getpid()} 下载模型完成，环境: {accelerator}。")

                # 释放锁并结束循环
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                break

            except IOError:
                logger.info(f"进程 {os.getpid()} 等待锁中...")
                time.sleep(1)  # 等待后重试
