import traceback
from typing import Any, List
from fastapi import APIRouter, Body, File, UploadFile, Form
import uuid
import os
import shutil
import pandas as pd
from pairag.core.models.errors import ServiceError, UserInputError
from pairag.data_pipeline.job.rag_job_manager import job_manager
from pairag.knowledgebase.rag_knowledgebase import KnowledgeBase, knowledgebase_manager
from pairag.core.chat_service import chat_service
from pairag.core.rag_module import resolve
from pairag.chat.models import RetrievalRequest
from fastapi.responses import StreamingResponse
from loguru import logger

from pairag.integrations.data_analysis.text2sql.utils.constants import (
    DEFAULT_DESCRIPTION_FOLDER_PATH,
    DEFAULT_DB_HISTORY_PATH,
    DEFAULT_DB_HISTORY_NAME,
)
from pairag.utils.constants import DEFAULT_KNOWLEDGEBASE_PATH
from pairag.file.store.oss_store import PaiOssStore

v1_router = APIRouter()


@v1_router.patch("/config")
async def aupdate(new_config: Any = Body(None)):
    chat_service.reload(new_config)
    return {"msg": "Update RAG configuration successfully."}


@v1_router.get("/config")
async def aconfig():
    return chat_service.get_config()


@v1_router.get("/indexes/{index_name}")
async def get_index(index_name: str):
    try:
        return knowledgebase_manager.get_knowledgebase(name=index_name)
    except Exception as ex:
        logger.error(f"Get index '{index_name}' failed: {ex} {traceback.format_exc()}")
        raise UserInputError(f"Get index '{index_name}' failed: {ex}")


@v1_router.post("/indexes/{index_name}")
async def add_index(index_name: str, index_entry: KnowledgeBase):
    try:
        knowledgebase_manager.add_knowledgebase(index_entry)
        return {"msg": f"Add index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(f"Add index '{index_name}' failed: {ex} {traceback.format_exc()}")
        raise UserInputError(f"Add index '{index_name}' failed: {ex}")


@v1_router.patch("/indexes/{index_name}")
async def update_index(index_name: str, index_entry: KnowledgeBase):
    try:
        knowledgebase_manager.update_knowledgebase(index_entry)
        return {"msg": f"Update index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Update index '{index_name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Update index '{index_name}' failed: {ex}")


@v1_router.delete("/indexes/{index_name}")
async def delete_index(index_name: str):
    try:
        knowledgebase_manager.delete_knowledgebase(index_name)
        return {"msg": f"Delete index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Delete index '{index_name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Delete index '{index_name}' failed: {ex}")


@v1_router.get("/indexes")
async def list_indexes():
    return knowledgebase_manager.list_knowledgebases()


# New knowledgebase API


@v1_router.get("/knowledgebases/{name}")
async def get_knowledgebase(name: str):
    """查询指定知识库信息"""
    try:
        return knowledgebase_manager.get_knowledgebase(name=name)
    except Exception as ex:
        logger.error(
            f"Get knowledgebase '{name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Get knowledgebase '{name}' failed: {ex}")


@v1_router.post("/knowledgebases/{name}")
async def add_knowledgebase(name: str, knowledgebase: KnowledgeBase):
    """新增知识库"""
    try:
        knowledgebase_manager.add_knowledgebase(knowledgebase)
        return {"msg": f"Add knowledgebase '{name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Add knowledgebase '{name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Add knowledgebase '{name}' failed: {ex}")


@v1_router.patch("/knowledgebases/{name}")
async def update_knowledgebase(name: str, knowledgebase: KnowledgeBase):
    """更新指定知识库"""
    try:
        knowledgebase_manager.update_knowledgebase(knowledgebase)
        return {"msg": f"Update knowledgebase '{name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Update knowledgebase '{name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Update knowledgebase '{name}' failed: {ex}")


@v1_router.delete("/knowledgebases/{name}")
async def delete_knowledgebase(name: str):
    """删除指定知识库"""
    try:
        knowledgebase_manager.delete_knowledgebase(name)
        return {"msg": f"Delete knowledgebase '{name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Delete knowledgebase '{name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Delete knowledgebase '{name}' failed: {ex}")


@v1_router.get("/knowledgebases")
async def list_knowledgebases():
    """知识库列表"""
    return knowledgebase_manager.list_knowledgebases()


@v1_router.get("/knowledgebases/{name}/files")
async def list_knowledgebase_files(name: str):
    """指定知识库查询文件列表"""
    return knowledgebase_manager.get_docs_from_knowledgebase(name)


@v1_router.get("/knowledgebases/{name}/history")
async def get_upload_history(name: str):
    """新知识库查询上传历史"""
    return job_manager.get_job_history(name)


@v1_router.post("/knowledgebases/{name}/files")
async def add_file_to_knowledgebase(name: str, files: List[UploadFile] = File(...)):
    """新知识库上传文件"""
    if name not in knowledgebase_manager._knowledgebase_map.knowledgebases:
        raise UserInputError(f"Knowledgebase '{name}' not found.")

    if not files:
        raise UserInputError("No files provided.")

    knowledge_docs_dir = os.path.join(DEFAULT_KNOWLEDGEBASE_PATH, name, "docs")
    os.makedirs(knowledge_docs_dir, exist_ok=True)

    for file in files:
        file_name = file.filename
        file_data = await file.read()
        save_file_name = os.path.join(
            DEFAULT_KNOWLEDGEBASE_PATH,
            name,
            "docs",
            file_name,
        )
        with open(save_file_name, "wb") as f:
            f.write(file_data)
        logger.info(f"File {file_name} has been save to {save_file_name}.")

    return {"message": "Files have been successfully uploaded."}


@v1_router.post("/knowledgebases/{name}/oss_files")
async def add_oss_file_to_knowledgebase(
    name: str,
    files: List[str] = Form(...),
    oss_bucket_name: str = Form(...),
    oss_endpoint: str = Form(...),
):
    """新知识库上传文件"""
    """"
    Example:
    curl -X "POST" http://127.0.0.1:8687/api/v1/knowledgebases/INDEX_2/oss_files
    -F 'files=oss://pai-rag/files/file1.pdf'
    -F 'files=oss://pai-rag/files/file2.pdf'
    -F 'oss_bucket_name=pai-rag'
    -F 'oss_endpoint=oss-cn-hangzhou.aliyuncs.com'
    """
    if name not in knowledgebase_manager._knowledgebase_map.knowledgebases:
        raise UserInputError(f"Knowledgebase '{name}' not found.")

    if not files:
        raise UserInputError("Oss_files not provided.")

    knowledge_docs_dir = os.path.join(DEFAULT_KNOWLEDGEBASE_PATH, name, "docs")
    os.makedirs(knowledge_docs_dir, exist_ok=True)

    oss_store = None
    if oss_bucket_name and oss_endpoint:
        oss_store = resolve(
            cls=PaiOssStore,
            bucket_name=oss_bucket_name,
            endpoint=oss_endpoint,
        )
    else:
        raise UserInputError("Either oss_bucket_name or oss_endpoint is not provided.")

    local_path = os.path.join(
        DEFAULT_KNOWLEDGEBASE_PATH,
        name,
        "docs",
    )
    for oss_file_path in files:
        oss_file_name = oss_file_path.split("/")[-1]
        local_files = oss_store.download_oss_file_to_local(
            oss_path=oss_file_path, local_path=local_path, oss_store=oss_store
        )
        logger.info(f"Oss file {oss_file_name} has been save to {local_files}.")

    return {"message": "Oss files have been successfully uploaded."}


@v1_router.get("/knowledgebases/{name}/files/{file_name}")
async def get_file_from_knowledgebase(name: str, file_name: str):
    """新知识库查询文件上传状态"""
    if name not in knowledgebase_manager._knowledgebase_map.knowledgebases:
        raise UserInputError(f"Knowledgebase '{name}' not found.")

    if not file_name:
        raise UserInputError(f"file_name '{file_name}' cannot be empty.")

    return job_manager.get_file_upload_status(name, file_name)


@v1_router.delete("/knowledgebases/{name}/files/{file_name}")
async def delete_file_from_knowledgebase(name: str, file_name: str):
    """新知识库删除文件"""
    if name not in knowledgebase_manager._knowledgebase_map.knowledgebases:
        raise UserInputError(f"Knowledgebase '{name}' not found.")

    if not file_name:
        raise UserInputError(f"file_name '{file_name}' cannot be empty.")

    save_file_name = os.path.join(
        DEFAULT_KNOWLEDGEBASE_PATH,
        name,
        "docs",
        file_name,
    )
    if not os.path.exists(save_file_name):
        raise UserInputError(f"File '{file_name}' not found")

    if os.path.isdir(save_file_name):
        raise UserInputError(f"Deleting a directory '{file_name}' is not supported.")

    try:
        os.remove(save_file_name)
        return {"message": f"File '{file_name}' have been successfully removed."}
    except Exception as e:
        raise ServiceError(f"Error deleting file '{file_name}': {str(e)}")


@v1_router.post("/retrieval")
async def aknowledgebase_retrieval(retrieval_request: RetrievalRequest):
    response = await chat_service.aknowledgebase_retrieval(retrieval_request)
    return response


@v1_router.post("/upload_datasheet")
async def upload_datasheet(
    file: UploadFile,
):
    task_id = uuid.uuid4().hex
    if not file:
        return None

    persist_path = "./localdata/data_analysis"

    os.makedirs(name=persist_path, exist_ok=True)

    # 清空目录中的文件
    for filename in os.listdir(persist_path):
        file_path = os.path.join(persist_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.info(f"Failed to delete {file_path}. Reason: {e}")

    # 指定持久化存储位置
    file_name = os.path.basename(file.filename)  # 获取文件名
    destination_path = os.path.join(persist_path, file_name)
    # 写入文件
    try:
        # shutil.copy(file.filename, destination_path)
        with open(destination_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("data analysis file saved successfully")

        if destination_path.endswith(".csv"):
            df = pd.read_csv(destination_path)
        elif destination_path.endswith(".xlsx"):
            df = pd.read_excel(destination_path)
        else:
            raise TypeError("Unsupported file type.")

    except Exception as e:
        return StreamingResponse(status_code=500, content={"message": str(e)})

    return {
        "task_id": task_id,
        "destination_path": destination_path,
        "data_preview": df.head(10).to_json(orient="records", lines=False),
    }


@v1_router.post("/upload_db_history")
async def upload_history_json(
    file: UploadFile,
    db_name: str = Form(None),
):
    task_id = uuid.uuid4().hex
    if not file:
        return None

    persist_path = DEFAULT_DB_HISTORY_PATH

    os.makedirs(name=persist_path, exist_ok=True)

    # # 清空目录中的文件
    # for filename in os.listdir(persist_path):
    #     file_path = os.path.join(persist_path, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #     except Exception as e:
    #         logger.info(f"Failed to delete {file_path}. Reason: {e}")

    # 指定持久化存储位置
    file_name = os.path.basename(file.filename)  # 获取文件名
    destination_path = os.path.join(persist_path, f"{db_name}_{file_name}")
    # 写入文件
    try:
        # shutil.copy(file.filename, destination_path)
        with open(destination_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("History file saved successfully")

    except Exception as e:
        return StreamingResponse(status_code=500, content={"message": str(e)})

    # 重命名
    try:
        unified_destination_path = os.path.join(
            persist_path, f"{db_name}_{DEFAULT_DB_HISTORY_NAME}"
        )
        os.rename(destination_path, unified_destination_path)
        logger.info("History file renamed successfully")
    except Exception as e:
        return StreamingResponse(status_code=500, content={"message": str(e)})

    return {
        "task_id": task_id,
        "destination_path": unified_destination_path,
    }


@v1_router.post("/upload_db_description")
async def upload_description(
    files: List[UploadFile] = Body(None),
    db_name: str = Form(None),
):
    task_id = uuid.uuid4().hex
    if not files:
        return {"message": "No upload files"}

    persist_path = DEFAULT_DESCRIPTION_FOLDER_PATH
    file_destination_folder = os.path.join(
        persist_path, db_name, "database_description"
    )
    os.makedirs(name=file_destination_folder, exist_ok=True)

    # 指定持久化存储位置
    for file in files:
        file_name = os.path.basename(file.filename)  # 获取文件名
        file_destination_path = os.path.join(
            persist_path, db_name, "database_description", file_name
        )
        # 写入文件
        try:
            # shutil.copy(file.filename, destination_path)
            with open(file_destination_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            logger.info("History file saved successfully")

        except Exception as e:
            return StreamingResponse(status_code=500, content={"message": str(e)})

    return {
        "task_id": task_id,
        "destination_path": persist_path,
    }


@v1_router.post("/query/load_db_info")
async def aload_db_info():
    task_id = uuid.uuid4().hex
    await chat_service.aload_db_info()
    return {"task_id": task_id}


@v1_router.get("/health")
def health_check():
    return {"status": "OK"}
