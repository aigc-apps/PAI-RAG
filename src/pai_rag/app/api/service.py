from fastapi import FastAPI
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api import query
from pai_rag.app.api.v1.chat import router_v1
from pai_rag.app.api.v1.openai_chat import router_openai
from pai_rag.app.api import agent_demo
from pai_rag.app.api.middleware import init_middleware
from pai_rag.app.api.error_handler import config_app_errors
from pai_rag.app.web.webui import configure_webapp
from pai_rag.core.rag_environment import service_environment


def init_router(app: FastAPI):
    # app.include_router(base_router.router, prefix="", tags=["base"])
    app.include_router(
        query.router, prefix="/service", tags=["RAG_forward_compatibility"]
    )
    app.include_router(router_openai, prefix="/v1", tags=["openai_compatible"])
    app.include_router(router_v1, prefix="/api/v1", tags=["api_v1"])
    app.include_router(agent_demo.demo_router, tags=["AgentDemo"], prefix="/demo/api")
    if service_environment.IS_MULTIPLE_INSTANCE:
        from pai_rag.app.api.v1.home import router_home

        app.include_router(router_home, prefix="/", tags=["Homepage"])
    else:
        configure_webapp(app)


def configure_app(app: FastAPI):
    rag_service.initialize()
    init_middleware(app)
    init_router(app)
    config_app_errors(app)
