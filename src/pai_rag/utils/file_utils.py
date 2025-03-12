import hashlib
from loguru import logger


def generate_md5(file_path):
    """Generate MD5 hash of the content of the specified file."""
    try:
        # Read the file in chunks to avoid using too much memory
        with open(file_path, "rb") as file:
            md5 = hashlib.md5()  # Create an MD5 hash object

            while chunk := file.read(8192):  # Read the file in 8 KB chunks
                md5.update(chunk)  # Update the hash with the chunk

        return md5.hexdigest()  # Return the hexadecimal representation of the hash
    except Exception as ex:
        logger.error(f"Error generating md5 for file '{file_path}'.")
        raise ex
