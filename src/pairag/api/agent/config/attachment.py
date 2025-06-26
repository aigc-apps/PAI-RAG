### Embedding configuration API ###

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pairag.api.response_model import error_response
from loguru import logger
import uuid

attachments_router = APIRouter()


@attachments_router.post("/upload")
async def upload_attachment_file(
    file_id: str = Form(...), file: UploadFile = File(...)
):
    # 文件大小限制
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    if file.size > MAX_SIZE:
        return JSONResponse(
            status_code=413,
            content=error_response(code=413, message="文件大小超过限制"),
        )

    # 生成唯一文件名
    local_file_id = str(uuid.uuid4())
    file_path = f"localdata/attachments/{file_id}_{file.filename}"
    logger.info(f"Saving file: {file_path}")
    # 保存文件
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            logger.info("File saved.")
        logger.info("Return JSONResponse")
        return JSONResponse(
            status_code=200,
            content={
                "id": local_file_id,
                "fid": file_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size,
                "status": "success",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(code=500, message=str(e)),
        )
