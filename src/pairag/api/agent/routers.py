from fastapi import FastAPI


def add_config_router(app: FastAPI):
    from pairag.api.agent.config.llm import llm_router
    from pairag.api.agent.config.mcp_server import mcp_router
    from pairag.api.agent.config.websearch import websearch_router
    from pairag.api.agent.config.trace import trace_router
    from pairag.api.agent.config.embedding import embedding_router
    from pairag.api.agent.config.knowledgebase import knowledgebase_router
    from pairag.api.agent.config.attachment import attachments_router

    app.include_router(llm_router, prefix="/v1/config/llms")
    app.include_router(mcp_router, prefix="/v1/config/mcps")
    app.include_router(websearch_router, prefix="/v1/config/websearch")
    app.include_router(trace_router, prefix="/v1/config/trace")
    app.include_router(embedding_router, prefix="/v1/config/embeddings")
    app.include_router(knowledgebase_router, prefix="/v1/config/knowledgebases")
    app.include_router(attachments_router, prefix="/v1/config/attachments")


def add_chat_router(app: FastAPI):
    from pairag.api.agent.chat import chat_agent_router

    app.include_router(chat_agent_router, prefix="/v1/agent/chat")
