from fastapi import FastAPI


def add_config_router(app: FastAPI):
    from api.v1.config_apis.llm import llm_router
    from api.v1.config_apis.mcp_server import mcp_router
    from api.v1.config_apis.websearch import websearch_router
    from api.v1.config_apis.trace import trace_router
    from api.v1.config_apis.embedding import embedding_router
    from api.v1.config_apis.attachment import attachments_router
    from api.v1.config_apis.reranker import reranker_router
    from api.v1.config_apis.metadata import knowledgebase_router
    from api.v1.config_apis.chatapp import app_router
    from api.v1.config_apis.role.role import role_router
    from api.v1.config_apis.guardrail import guardrail_router

    app.include_router(llm_router, prefix="/v1/config/llms")
    app.include_router(mcp_router, prefix="/v1/config/mcps")
    app.include_router(websearch_router, prefix="/v1/config/websearch")
    app.include_router(trace_router, prefix="/v1/config/trace")
    app.include_router(embedding_router, prefix="/v1/config/embeddings")
    app.include_router(reranker_router, prefix="/v1/config/rerankers")
    app.include_router(knowledgebase_router, prefix="/v1/config/knowledgebases")
    app.include_router(attachments_router, prefix="/v1/config/attachments")
    app.include_router(app_router, prefix="/v1/config/apps")
    app.include_router(role_router, prefix="/v1/config/roles")
    app.include_router(guardrail_router, prefix="/v1/config/guardrail")


def add_chat_router(app: FastAPI):
    from api.v1.chat import chat_agent_router
    from api.v1.thread import thread_router
    from api.v1.retrieval import retrieval_router
    from api.v1.healthcheck import health_router
    from api.v1.embed import embedding_router

    app.include_router(chat_agent_router, prefix="/v1/chat/completions")
    app.include_router(thread_router, prefix="/v1/threads")
    app.include_router(retrieval_router, prefix="/v1/retrieval")
    app.include_router(health_router, prefix="/v1/health")
    app.include_router(embedding_router, prefix="/v1/embeddings")
