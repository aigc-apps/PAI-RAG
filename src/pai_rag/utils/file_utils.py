import hashlib
import logging
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
)
import os


def get_modified_time(file_path):
    state = os.stat(file_path)
    return state.st_mtime


def generate_text_md5(text):
    text_md5 = hashlib.md5()  # Create an MD5 hash object
    # Encode the file path string to bytes and update the hash
    text_md5.update(text.encode("utf-8"))
    return text_md5.hexdigest()


# 读取文件的retry机制
@retry(
    wait=wait_fixed(1),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(OSError),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def generate_file_md5(file_path):
    with open(file_path, "rb") as file:
        file_content_md5 = hashlib.md5()  # Create an MD5 hash object

        while chunk := file.read(8192):  # Read the file in 8 KB chunks
            file_content_md5.update(chunk)  # Update the hash with the chunk

        return file_content_md5.hexdigest()


def generate_md5(file_path):
    """Generate MD5 hash of the content of the specified file."""
    try:
        return (
            generate_text_md5(file_path),
            generate_file_md5(file_path),
        )  # Return the hexadecimal representation of the hash
    except Exception as ex:
        logger.error(f"Error generating md5 for file '{file_path}'.")
        raise ex
