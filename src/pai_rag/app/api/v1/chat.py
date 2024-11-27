import traceback
from typing import Any, List
from fastapi import APIRouter, Body, BackgroundTasks, UploadFile, Form
import uuid
import hashlib
import os
import json
import tempfile
import shutil
import pandas as pd
from pai_rag.core.models.errors import UserInputError
from pai_rag.core.rag_index_manager import RagIndexEntry, index_manager
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api.models import (
    RagQuery,
    RetrievalQuery,
)
from fastapi.responses import StreamingResponse
from loguru import logger

from pai_rag.integrations.nodeparsers.pai.pai_node_parser import (
    COMMON_FILE_PATH_FODER_NAME,
)

router_v1 = APIRouter()


@router_v1.post("/query")
async def aquery_v1(query: RagQuery):
    response = await rag_service.aquery_v1(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_v1.post("/query/llm")
async def aquery_llm_v1(query: RagQuery):
    response = await rag_service.aquery_llm_v1(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_v1.post("/query/search")
async def aquery_search_v1(query: RagQuery):
    response = await rag_service.aquery_search_v1(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_v1.post("/query/retrieval")
async def aquery_retrieval(query: RetrievalQuery):
    return await rag_service.aquery_retrieval(query)


@router_v1.post("/query/agent")
async def aquery_agent(query: RagQuery):
    response = await rag_service.aquery_agent_v1(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_v1.post("/config/agent")
async def aload_agent_config(file: UploadFile):
    fn = file.filename
    data = await file.read()
    file_hash = hashlib.md5(data).hexdigest()
    save_file = os.path.join("localdata", f"{file_hash}_{fn}")

    with open(save_file, "wb") as f:
        f.write(data)
        f.close()
    return await rag_service.aload_agent_config(save_file)


@router_v1.patch("/config")
async def aupdate(new_config: Any = Body(None)):
    rag_service.reload(new_config)
    return {"msg": "Update RAG configuration successfully."}


@router_v1.get("/config")
async def aconfig():
    return rag_service.get_config()


@router_v1.get("/indexes/{index_name}")
async def get_index(index_name: str):
    try:
        return index_manager.get_index_by_name(index_name=index_name)
    except Exception as ex:
        logger.error(f"Get index '{index_name}' failed: {ex} {traceback.format_exc()}")
        raise UserInputError(f"Get index '{index_name}' failed: {ex}")


@router_v1.post("/indexes/{index_name}")
async def add_index(index_name: str, index_entry: RagIndexEntry):
    try:
        index_manager.add_index(index_entry)
        return {"msg": f"Add index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(f"Add index '{index_name}' failed: {ex} {traceback.format_exc()}")
        raise UserInputError(f"Add index '{index_name}' failed: {ex}")


@router_v1.patch("/indexes/{index_name}")
async def update_index(index_name: str, index_entry: RagIndexEntry):
    try:
        index_manager.update_index(index_entry)
        return {"msg": f"Update index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Update index '{index_name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Update index '{index_name}' failed: {ex}")


@router_v1.delete("/indexes/{index_name}")
async def delete_index(index_name: str):
    try:
        index_manager.delete_index(index_name)
        return {"msg": f"Delete index '{index_name}' successfully."}
    except Exception as ex:
        logger.error(
            f"Delete index '{index_name}' failed: {ex} {traceback.format_exc()}"
        )
        raise UserInputError(f"Delete index '{index_name}' failed: {ex}")


@router_v1.get("/indexes")
async def list_indexes():
    return index_manager.list_indexes()


@router_v1.get("/get_upload_state")
def task_status(task_id: str):
    status, detail = rag_service.get_task_status(task_id)
    return {"task_id": task_id, "status": status, "detail": detail}


@router_v1.post("/upload_data")
async def upload_data(
    files: List[UploadFile] = Body(None),
    oss_path: str = Form(None),
    index_name: str = Form(None),
    enable_raptor: bool = Form(False),
    enable_multimodal: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    task_id = uuid.uuid4().hex
    logger.info(
        f"Upload data task_id: {task_id} index_name: {index_name} enable_multimodal: {enable_multimodal}"
    )
    if oss_path:
        background_tasks.add_task(
            rag_service.add_knowledge,
            task_id=task_id,
            filter_pattern=None,
            oss_path=oss_path,
            from_oss=True,
            index_name=index_name,
            enable_raptor=enable_raptor,
            enable_multimodal=enable_multimodal,
        )
    else:
        if not files:
            return {"message": "No upload file sent"}
        tmpdir = tempfile.mkdtemp()
        input_files = []
        for file in files:
            fn = file.filename
            data = await file.read()
            file_hash = hashlib.md5(data).hexdigest()
            tmp_file_dir = os.path.join(
                tmpdir, f"{COMMON_FILE_PATH_FODER_NAME}/{file_hash}"
            )
            os.makedirs(tmp_file_dir, exist_ok=True)
            save_file = os.path.join(tmp_file_dir, fn)

            with open(save_file, "wb") as f:
                f.write(data)
                f.close()
            input_files.append(save_file)

        background_tasks.add_task(
            rag_service.add_knowledge,
            task_id=task_id,
            input_files=input_files,
            filter_pattern=None,
            index_name=index_name,
            oss_path=None,
            enable_raptor=enable_raptor,
            temp_file_dir=tmpdir,
            enable_multimodal=enable_multimodal,
        )

    return {"task_id": task_id}


@router_v1.post("/upload_datasheet")
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


@router_v1.post("/query/data_analysis")
async def aquery_analysis(query: RagQuery):
    response = await rag_service.aquery_analysis_v1(query)
    if not query.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_v1.post("/query/custom_search")
async def aquery_custom_test(query: RagQuery):
    try:
        response = await rag_service.aquery_llm(query)

        try:
            answer = json.loads(response.answer)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return {
                "status": "error",
                "message": "Parsing Error: The LLM response is not a valid JSON format.",
                "status_code": 400,
            }

        input_list = [res.get("型号") for res in answer if "型号" in res]
        logger.info(f"Extracted input list: {input_list}")
        if not input_list:
            logger.warning("No model information found in response.")
            return {
                "status": "error",
                "message": "Parsing Error: The '型号' key is not found in the JSON.",
                "status_code": 404,
            }

        unique_input_list = list(set(input_list))
        logger.info(f"Unique input list: {unique_input_list}")

        try:
            sql_response = rag_service.sql_query(unique_input_list)
            return {
                "status": "success",
                "data": {
                    "input": unique_input_list,
                    "output": sql_response,
                },
                "status_code": 200,
            }
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return {
                "status": "error",
                "message": "SQL query failed: No information found for the relevant input list.",
                "status_code": 500,
            }

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "status": "error",
            "message": "Unexpected error, please try again later.",
            "status_code": 500,
        }
