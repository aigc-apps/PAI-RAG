from typing import Any
from fastapi import APIRouter, Body
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api.models import (
    RagQuery,
    LlmQuery,
    RetrievalQuery,
    RagResponse,
    LlmResponse,
    KnowledgeInput,
)

router = APIRouter()


@router.post("/query")
async def aquery(query: RagQuery) -> RagResponse:
    return await rag_service.aquery(query)


@router.post("/query/llm")
async def aquery_llm(query: LlmQuery) -> LlmResponse:
    return await rag_service.aquery_llm(query)


@router.post("/query/retrieval")
async def aquery_retrieval(query: RetrievalQuery):
    return await rag_service.aquery_vectordb(query)

@router.post("/query/agent")
async def aquery_agent(query: LlmQuery) -> LlmResponse:
    return await rag_service.aquery_agent(query)

@router.patch("/config")
async def aupdate(new_config: Any = Body(None)):
    rag_service.reload(new_config)
    return {"msg": "Update RAG configuration successfully."}


@router.post("/knowledge")
async def load_knowledge(input: KnowledgeInput):
    await rag_service.add_knowledge(
        file_dir=input.file_path, enable_qa_extraction=input.enable_qa_extraction
    )
    return {"msg": "Update RAG configuration successfully."}


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
