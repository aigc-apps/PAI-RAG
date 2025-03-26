import os
from typing import List
import pathlib
from pai_rag.integrations.readers.pai.constants import ACCEPTABLE_DOC_TYPES
from loguru import logger


def get_input_files(
    file_path_or_directory: str | List[str],
    filter_pattern: str = None,
):
    filter_pattern = filter_pattern or "*"

    input_files = None
    if isinstance(file_path_or_directory, list):
        # file list
        input_files = [
            f
            for f in file_path_or_directory
            if os.path.isfile(f)
            and pathlib.Path(f).suffix.lower() in ACCEPTABLE_DOC_TYPES
        ]
    elif isinstance(file_path_or_directory, str) and os.path.isdir(
        file_path_or_directory
    ):
        # glob from directory
        directory = pathlib.Path(file_path_or_directory)
        input_files = [
            f
            for f in directory.rglob(filter_pattern)
            if os.path.isfile(f)
            and pathlib.Path(f).suffix.lower() in ACCEPTABLE_DOC_TYPES
        ]
    elif pathlib.Path(file_path_or_directory).suffix.lower() in ACCEPTABLE_DOC_TYPES:
        # Single file
        input_files = [pathlib.Path(file_path_or_directory)]
    else:
        raise ValueError(
            f"Invalid input path or not supported file type for '{file_path_or_directory}'."
        )

    if not input_files:
        raise ValueError(
            f"No file found at path '{file_path_or_directory}' with pattern '{filter_pattern}'."
        )

    logger.info(
        f"Found {len(input_files)} files at path '{file_path_or_directory}' with pattern '{filter_pattern}'. Samples: {input_files[:5]}"
    )
    return input_files
