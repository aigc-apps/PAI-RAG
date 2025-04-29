from typing import List


def parse_file_extensions(file_extensions_str: str) -> List[str]:
    """
    Parse the file extensions string into a list of file extensions.

    Args:
        file_extensions_str (str): The file extensions string.

    Returns:
        List[str]: The list of file extensions.
    """
    file_extensions = file_extensions_str.split(",")
    extensions = [ext.strip() for ext in file_extensions if ext.strip()]
    return extensions