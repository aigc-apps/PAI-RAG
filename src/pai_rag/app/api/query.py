from typing import Any
from fastapi import APIRouter, Body, BackgroundTasks
import uuid
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api.models import (
    RagQuery,
    LlmQuery,
    RetrievalQuery,
    RagResponse,
    LlmResponse,
    DataInput,
)

router = APIRouter()

upload_tasks = {}


@router.post("/query")
async def aquery(query: RagQuery) -> RagResponse:
    return await rag_service.aquery(query)


@router.post("/query/llm")
async def aquery_llm(query: LlmQuery) -> LlmResponse:
    return await rag_service.aquery_llm(query)


@router.post("/query/retrieval")
async def aquery_retrieval(query: RetrievalQuery):
    return await rag_service.aquery_retrieval(query)


@router.post("/query/agent")
async def aquery_agent(query: LlmQuery) -> LlmResponse:
    return await rag_service.aquery_agent(query)


@router.patch("/config")
async def aupdate(new_config: Any = Body(None)):
    rag_service.reload(new_config)
    return {"msg": "Update RAG configuration successfully."}


tasks_status = {}


@router.post("/upload_data")
async def load_data(input: DataInput, background_tasks: BackgroundTasks):
    task_id = uuid.uuid4().hex  # 生成唯一任务ID
    tasks_status[task_id] = "processing"
    # 添加后台任务并立即返回任务ID
    background_tasks.add_task(process_knowledge, task_id, input)
    return {"task_id": task_id}


def process_knowledge(task_id: str, input: DataInput):
    try:
        rag_service.add_knowledge(
            file_dir=input.file_path, enable_qa_extraction=input.enable_qa_extraction
        )
        tasks_status[task_id] = "completed"
    except Exception:
        tasks_status[task_id] = "failed"


@router.get("/get_upload_state")
def task_status(task_id: str):
    status = tasks_status.get(task_id, "unknown")
    return {"task_id": task_id, "status": status}


@router.post("/evaluate/response")
def evaluate_reponse():
    eval_results = rag_service.evaluate_reponse()
    return {"status": 200, "result": eval_results}


@router.post("/batch_evaluate/retrieval")
async def batch_retrieval_evaluate():
    df, eval_results = await rag_service.batch_evaluate_retrieval_and_response(
        type="retrieval"
    )
    return {"status": 200, "result": eval_results}


@router.post("/batch_evaluate/response")
async def batch_response_evaluate():
    df, eval_results = await rag_service.batch_evaluate_retrieval_and_response(
        type="response"
    )
    return {"status": 200, "result": eval_results}


@router.post("/batch_evaluate")
async def batch_evaluate():
    df, eval_results = await rag_service.batch_evaluate_retrieval_and_response(
        type="all"
    )
    return {"status": 200, "result": eval_results}
