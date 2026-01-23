import os
import aiohttp
import hashlib
from pydantic import BaseModel
from loguru import logger
from utils.http_session import HttpSessionShared

BACKEND_PORT = os.environ.get("BACKEND_PORT", "8682")
ATTACHMENT_UPLOAD_API = f"http://127.0.0.1:{BACKEND_PORT}/v1/config/attachments"
GAIA_ATTACHMENT_FOLDER = "./resources/dataset/gaia/attachments"

CONTENT_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".heic": "image/heic",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

class AttachmentFile(BaseModel):
    id: str
    name: str
    contentType: str = "text/plain"


async def upload_gaia_attachment_file(
    file_name: str,
    tenant_id: str = None,
):
    file_id = hashlib.md5(file_name.encode()).hexdigest()
    file_path = os.path.join(GAIA_ATTACHMENT_FOLDER, file_name)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("file_id", file_id)
            form.add_field(
                "file",
                f.read(),
                filename=file_name,
            )
            headers = {
                "X-TENANT-ID": tenant_id,
            }
            session = await HttpSessionShared.ensure_session()
            async with session.post(ATTACHMENT_UPLOAD_API, data=form, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Uploaded {file_name} successfully.")
                    result = await response.json()
                    return AttachmentFile(
                        id=result["data"]["id"],
                        name=result["data"]["file_name"],
                        contentType=CONTENT_TYPE_MAP.get(
                            result["data"]["file_extension"], "text/plain"
                        ),
                    )
                else:
                    logger.info(f"Failed to upload {file_name}")
                    raise Exception(f"Failed to upload {file_name}")
    else:
        raise FileNotFoundError(f"File {file_name} with path {file_path} not found.")
