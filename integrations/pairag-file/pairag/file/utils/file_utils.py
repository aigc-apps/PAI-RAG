import os


SUPPORTED_FILE_EXT = [
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".doc",
    ".ppt",
    ".pptx",
    ".csv",
    ".xlsx",
    ".xls",
    ".html",
    ".jsonl",
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".mkv",
]


def ensure_file_name_is_supported(filename: str):
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in SUPPORTED_FILE_EXT:
        raise ValueError(
            f"不支持的文件格式: {file_extension}. 请选择以下格式的文件: {SUPPORTED_FILE_EXT}"
        )


def ensure_file_type_is_supported(file_extension: str):
    if file_extension not in SUPPORTED_FILE_EXT:
        raise ValueError(
            f"不支持的文件格式: {file_extension}. 请选择以下格式的文件: {SUPPORTED_FILE_EXT}"
        )
