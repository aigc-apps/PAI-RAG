### Embedding configuration API ###

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pairag.mcp.online_file_readers.pai_online_data_reader import PaiOnlineDataReader
from pairag.mcp.online_file_readers.constants import ONLINE_ACCEPTABLE_DOC_TYPES
from loguru import logger
import os

attachments_router = APIRouter()
ATTACHMENTS_DIR = "localdata/attachments"
ATTACHMENTS_TMP_DIR = "localdata/attachments/tmp"
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
os.makedirs(ATTACHMENTS_TMP_DIR, exist_ok=True)

data_reader = PaiOnlineDataReader()


@attachments_router.post("/upload")
async def upload_attachment_file(
    file_id: str = Form(...), file: UploadFile = File(...)
):
    # 获取文件扩展名
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ONLINE_ACCEPTABLE_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .txt, .pdf, .docx, .md are allowed.",
        )

    try:
        # 保存文件到临时目录
        temp_file_path = f"{ATTACHMENTS_TMP_DIR}/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
            logger.info(f"File {file.filename} saved to {temp_file_path}")

        documents = data_reader.load_data(file_path_or_directory=temp_file_path)

        logger.info(f"documents: {len(documents)} {documents}")

        # tmp process: 对解析后的文件直接存储到本地，以file_id命名

        # TODO:
        # 1. 文件内容存储到数据库，以file_id为主键
        # 2. 如果文件内容过长，需要进行截断存储
        # 3. 对截断的大文件进行分块和索引存储
        with open(f"{ATTACHMENTS_DIR}/{file_id}.txt", "wb") as f:
            f.write(documents[0].text.encode("utf-8"))
            logger.info(f"File {file_id}.txt saved to {ATTACHMENTS_DIR}")

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
