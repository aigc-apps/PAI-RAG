import os
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import with_async_db_session
from db.models.knowledgebase.file import KbFileEntity
import aiohttp
from pydantic import BaseModel
from loguru import logger
from utils.http_session import HttpSessionShared

BACKEND_PORT = os.environ.get("BACKEND_PORT", "8682")
ATTACHMENT_UPLOAD_API = f"http://127.0.0.1:{BACKEND_PORT}/v1/config/attachments"
GAIA_ATTACHMENT_FILES = "./resources/dataset/gaia/attachments"

CONTENT_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
class AttachmentFile(BaseModel):
    id: str
    name: str
    contentType: str = "text/plain"


@with_async_db_session
async def upload_gaia_attachment_file(
    session: AsyncSession, file_name: str
):
    file_id = file_name.split(".")[0]
    file_entity = await session.get(KbFileEntity, file_id)
    if file_entity:
        logger.info(f"File {file_id} already exists.")
        result = file_entity.model_dump()
        return AttachmentFile(
            id=result["id"],
            name=result["file_name"],
            contentType=CONTENT_TYPE_MAP.get(
                result["file_extension"], "text/plain"
            ),
        )
    else:
        file_path = os.path.join(GAIA_ATTACHMENT_FILES, file_name)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field("file_id", file_id)
                form.add_field(
                    "file",
                    f.read(),
                    filename=file_name,
                )

                session = await HttpSessionShared.ensure_session()
                async with session.post(ATTACHMENT_UPLOAD_API, data=form) as response:
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
