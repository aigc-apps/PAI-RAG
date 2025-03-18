import traceback
from typing import Any
from fastapi import APIRouter, Body, UploadFile
import uuid
import os
import shutil
import pandas as pd
from pai_rag.core.models.errors import UserInputError
from pai_rag.knowledgebase.rag_knowledgebase import KnowledgeBase, knowledgebase_manager
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api.models import RagQuery
from fastapi.responses import StreamingResponse
from loguru import logger


router = APIRouter()


@router.post("/query")
async def aquery(query: RagQuery):
    response = await rag_service.aquery(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router.post("/query/llm")
async def aquery_llm(query: RagQuery):
    response = await rag_service.aquery_llm(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router.post("/query/search")
async def aquery_search(query: RagQuery):
    response = await rag_service.aquery_search(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router.post("/query/retrieval")
async def aquery_retrieval(query: RagQuery):
    return await rag_service.aquery_retrieval(query)


@router.post("/query/agent")
async def aquery_agent(query: RagQuery):
    response = await rag_service.aquery_agent(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router.patch("/config")
async def aupdate(new_config: Any = Body(None)):
    rag_service.reload(new_config)
    return {"msg": "Update RAG configuration successfully."}


@router.get("/config")
async def aconfig():
    return rag_service.get_config()


@router.get("/indexes/{index_name}")
async def get_index(index_name: str):
    try:
        return knowledgebase_manager.get_knowledgebase(name=index_name)
    except Exception as ex:
        logger.error(f"Get index '{index_name}' failed: {ex} {traceback.format_exc()}")
        raise UserInputError(f"Get index '{index_name}' failed: {ex}")


@router.post("/indexes/{index_name}")
async def add_index(index_name: str, index_entry: KnowledgeBase):
    try:
        knowledgebase_manager.add_knowledgebase(index_entry)
        return {"msg": f"Add index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(f"Add index '{index_name}' failed: {ex} {traceback.format_exc()}")
        raise UserInputError(f"Add index '{index_name}' failed: {ex}")


@router.patch("/indexes/{index_name}")
async def update_index(index_name: str, index_entry: KnowledgeBase):
    try:
        knowledgebase_manager.update_knowledgebase(index_entry)
        return {"msg": f"Update index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Update index '{index_name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Update index '{index_name}' failed: {ex}")


@router.delete("/indexes/{index_name}")
async def delete_index(index_name: str):
    try:
        knowledgebase_manager.delete_knowledgebase(index_name)
        return {"msg": f"Delete index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Delete index '{index_name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Delete index '{index_name}' failed: {ex}")


@router.get("/indexes")
async def list_indexes():
    return knowledgebase_manager.list_knowledgebases()


@router.post("/upload_datasheet")
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


@router.post("/query/load_db_info")
async def aload_db_info():
    return await rag_service.aload_db_info()


@router.post("/query/data_analysis")
async def aquery_analysis(query: RagQuery):
    # await rag_service.aload_db_info()
    response = await rag_service.aquery_data_analysis_v1(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )
