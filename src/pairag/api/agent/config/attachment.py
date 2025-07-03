### Embedding configuration API ###

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from loguru import logger
import os
import docx2txt

attachments_router = APIRouter()
ATTACHMENTS_DIR = "localdata/attachments"
ATTACHMENTS_TMP_DIR = "localdata/attachments/tmp"
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
os.makedirs(ATTACHMENTS_TMP_DIR, exist_ok=True)


async def read_txt_file(file: UploadFile):
    content = await file.read()
    return content.decode("utf-8", errors="ignore")


async def read_docx_file(file: UploadFile):
    file_path = f"{ATTACHMENTS_TMP_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        content = docx2txt.process(file_path)
        return content
    finally:
        os.remove(file_path)


@attachments_router.post("/upload")
async def upload_attachment_file(
    file_id: str = Form(...), file: UploadFile = File(...)
):
    # 获取文件扩展名
    file_extension = os.path.splitext(file.filename)[1].lower()

    try:
        if file_extension == ".txt":
            content = await read_txt_file(file)
        elif file_extension == ".docx":
            content = await read_docx_file(file)
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Unsupported file format. Only .txt and .docx are supported."
                },
            )
        logger.info(f"File content: {content}")

        with open(f"{ATTACHMENTS_DIR}/{file_id}.txt", "wb") as f:
            f.write(
                content.encode("utf-8", errors="ignore")
                if isinstance(content, str)
                else content
            )

        return JSONResponse(
            status_code=200,
            content={
                "file_id": file_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "status": "success",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
