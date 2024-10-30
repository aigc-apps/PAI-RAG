from fastapi import FastAPI
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api import query
from pai_rag.app.api import agent_demo
from pai_rag.app.api.middleware import init_middleware
from pai_rag.app.api.error_handler import config_app_errors


def init_router(app: FastAPI):
    app.include_router(query.router, prefix="/service", tags=["RAG"])
    app.include_router(agent_demo.demo_router, tags=["AgentDemo"], prefix="/demo/api")


def configure_app(app: FastAPI, rag_configuration: RagConfigManager):
    rag_service.initialize(rag_configuration)
    init_middleware(app)
    init_router(app)
    config_app_errors(app)
