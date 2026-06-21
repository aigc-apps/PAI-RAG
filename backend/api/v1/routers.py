from fastapi import FastAPI


def add_config_router(app: FastAPI):
    from api.v1.config_apis.llm import llm_router
    from api.v1.config_apis.mcp_server import mcp_router
    from api.v1.config_apis.websearch import websearch_router
    from api.v1.config_apis.trace import trace_router
    from api.v1.config_apis.embedding import embedding_router
    from api.v1.config_apis.reranker import reranker_router
    from api.v1.config_apis.metadata import knowledgebase_router
    from api.v1.config_apis.datasource import datasource_router
    from api.v1.config_apis.chatapp import app_router
    from api.v1.config_apis.role.role import role_router
    from api.v1.config_apis.guardrail import guardrail_router
    from api.v1.config_apis.evaluation import evaluation_router
    from api.v1.config_apis.vectordb import vectordb_router

    from api.v1.config_apis.code_sandbox import code_sandbox_router
    from api.v1.config_apis.chatdb import chatdb_router

    from api.v1.fileview import fileview_router

    app.include_router(llm_router, prefix="/v1/config/llms")
    app.include_router(mcp_router, prefix="/v1/config/mcps")
    app.include_router(websearch_router, prefix="/v1/config/websearch")
    app.include_router(trace_router, prefix="/v1/config/trace")
    app.include_router(embedding_router, prefix="/v1/config/embeddings")
    app.include_router(reranker_router, prefix="/v1/config/rerankers")
    app.include_router(knowledgebase_router, prefix="/v1/config/knowledgebases")
    app.include_router(datasource_router, prefix="/v1/config/knowledgebases")
    app.include_router(app_router, prefix="/v1/config/apps")
    app.include_router(role_router, prefix="/v1/config/roles")
    app.include_router(guardrail_router, prefix="/v1/config/guardrail")
    app.include_router(evaluation_router, prefix="/v1/config/evaluation")
    app.include_router(vectordb_router, prefix="/v1/config/vectordb")

    app.include_router(code_sandbox_router, prefix="/v1/config/code_sandbox")
    app.include_router(chatdb_router, prefix="/v1/config/chatdb")
    app.include_router(fileview_router, prefix="/v1/fileview")


def add_chat_router(app: FastAPI):
    from api.v1.chat import chat_agent_router
    from api.v1.thread import thread_router
    from api.v1.retrieval import retrieval_router
    from api.v1.faq_retrieval import faq_retrieval_router
    from api.v1.retrieval_tool_api import retrieval_tool_router
    from api.v1.healthcheck import health_router
    from api.v1.embed import embedding_router
    from api.v1.files import files_router
    from api.v1.files_events import files_events_router
    from api.v1.files_uploads import files_uploads_router

    app.include_router(chat_agent_router, prefix="/v1/chat/completions")
    app.include_router(thread_router, prefix="/v1/threads")
    app.include_router(retrieval_router, prefix="/v1/retrieval")
    app.include_router(faq_retrieval_router, prefix="/v1/faq-retrieval")
    app.include_router(retrieval_tool_router, prefix="/v1/tools/retrieval")
    app.include_router(embedding_router, prefix="/v1/embeddings")
    # /events and /uploads must register BEFORE /v1/files so they aren't
    # shadowed by the `/v1/files/{file_id}` path parameter matcher.
    app.include_router(files_events_router, prefix="/v1/files/events")
    app.include_router(files_uploads_router, prefix="/v1/files/uploads")
    app.include_router(files_router, prefix="/v1/files")

    app.include_router(health_router, prefix="/health")
