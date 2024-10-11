from fastapi import APIRouter, FastAPI
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api import query
from pai_rag.app.api.middleware import init_middleware
from pai_rag.app.api.error_handler import config_app_errors


def init_router(app: FastAPI):
    api_router = APIRouter()
    api_router.include_router(query.router, tags=["RagQuery"])
    app.include_router(api_router, prefix="/service")


def configure_app(app: FastAPI, rag_configuration: RagConfiguration):
    rag_service.initialize(rag_configuration)
    init_middleware(app)
    init_router(app)
    config_app_errors(app)
