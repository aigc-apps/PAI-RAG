import base64
from typing import Protocol
from pairag.file.store.file_store_helper import file_store


class _FileLike(Protocol):
    """Duck-typed minimal surface for base64 encoding.

    Satisfied by both `FileEntity` (new, preferred) and anything else that
    exposes `file_path` / `tenant_id` / `file_extension`.
    """
    file_path: str
    tenant_id: str
    file_extension: str


mime_type_map = {
    # Images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".heic": "image/heic",
    # Videos
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    # Text
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".xml": "text/xml",
    ".csv": "text/csv",
    ".log": "text/plain",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".html": "text/html",
    ".css": "text/css",
    # Documents
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

multimodal_file_types = [
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
    ".webm",
]

def is_multimodal_file_type(file_extension: str) -> bool:
    if not file_extension:
        return False

    return file_extension.lower() in multimodal_file_types


def get_file_mime_type(file_extension: str) -> str:
    if not file_extension:
        return "application/octet-stream"

    return mime_type_map.get(file_extension.lower(), "application/octet-stream")


async def aget_file_base64_content(file: _FileLike) -> str:
    if not file.file_path:
        raise ValueError("File path is required")

    file_data = await file_store.read_async(file_path=file.file_path, tenant_id=file.tenant_id)
    if not file_data:
        raise ValueError("File content is required")

    file_data.seek(0)
    base64_data = base64.b64encode(file_data.read()).decode()
    mime_type = mime_type_map.get(file.file_extension, "application/octet-stream")
    return f"data:{mime_type};base64,{base64_data}"
