from fastapi import APIRouter, FastAPI
import gradio as gr
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api import query
from pai_rag.app.api.middleware import init_middleware
from pai_rag.app.web.webui import create_ui
from pai_rag.app.web.rag_client import rag_client

UI_PATH = ""


def init_router(app: FastAPI):
    api_router = APIRouter()
    api_router.include_router(query.router, tags=["RagQuery"])
    app.include_router(api_router, prefix="/service")


def configure_app(app: FastAPI, config_file: str, app_url: str):
    rag_service.initialize(config_file)

    init_middleware(app)
    init_router(app)

    rag_client.set_endpoint(app_url)
    ui = create_ui()
    ui.queue(concurrency_count=1, max_size=64)
    ui._queue.set_url(app_url)
    app = gr.mount_gradio_app(app, ui, path=UI_PATH)
    return app
