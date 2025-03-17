import hashlib
from loguru import logger


def generate_md5(file_path):
    """Generate MD5 hash of the content of the specified file."""
    try:
        # 1. file path to md5
        file_path_md5 = hashlib.md5()  # Create an MD5 hash object
        # Encode the file path string to bytes and update the hash
        file_path_md5.update(file_path.encode("utf-8"))

        # 2. file content to md5
        # Read the file in chunks to avoid using too much memory
        with open(file_path, "rb") as file:
            file_content_md5 = hashlib.md5()  # Create an MD5 hash object

            while chunk := file.read(8192):  # Read the file in 8 KB chunks
                file_content_md5.update(chunk)  # Update the hash with the chunk

        return (
            file_path_md5.hexdigest(),
            file_content_md5.hexdigest(),
        )  # Return the hexadecimal representation of the hash
    except Exception as ex:
        logger.error(f"Error generating md5 for file '{file_path}'.")
        raise ex
