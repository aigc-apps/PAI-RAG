import asyncio
from contextlib import asynccontextmanager
import click
import uvicorn
from fastapi import FastAPI
from pai_rag.app.api.service import configure_app
from pai_rag.app.web.webui import configure_webapp
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.core.rag_trace import init_trace
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.utils.constants import DEFAULT_MODEL_DIR, EAS_DEFAULT_MODEL_DIR
from logging.config import dictConfig
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


_BASE_DIR = Path(__file__).parent
_ROOT_BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")
DEFAULT_APPLICATION_EXAMPLE_DATA_FILE = os.path.join(
    _ROOT_BASE_DIR, "example_data/pai_document.pdf"
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_RAG_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/"
DEFAULT_GRADIO_PORT = 8002


def init_log():
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "correlation_id": {
                "()": "asgi_correlation_id.CorrelationIdFilter",
                "uuid_length": 32,
                "default_value": "-",
            },
        },
        "formatters": {
            "sample": {
                "format": "%(asctime)s %(levelname)s [%(correlation_id)s] %(message)s"
            },
            "verbose": {
                "format": "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s %(process)d %(thread)d %(message)s"
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s [%(correlation_id)s] - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "console": {
                "formatter": "verbose",
                "level": "DEBUG",
                "filters": ["correlation_id"],
                "class": "logging.StreamHandler",
            },
        },
        "loggers": {
            "": {"level": "INFO", "handlers": ["console"]},
        },
    }
    dictConfig(log_config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    from pai_rag.modules.index.index_daemon import index_daemon

    """Start all the non-blocking service tasks, which run in the background."""
    asyncio.create_task(index_daemon.refresh_async())
    yield


init_log()

app = FastAPI()

is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
if is_gunicorn:
    app = FastAPI(lifespan=lifespan)
    configure_app(app, DEFAULT_APPLICATION_CONFIG_FILE)


async def service_tasks_startup():
    from pai_rag.modules.index.index_daemon import index_daemon

    """Start all the non-blocking service tasks, which run in the background."""
    asyncio.create_task(index_daemon.refresh_async())


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("-V", "--version", is_flag=True, help="Show version and exit.")
def main(ctx, version):
    if version:
        click.echo(version)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option(
    "-h",
    "--host",
    show_default=True,
    help=f"WebApp Host IP. Default: {DEFAULT_HOST}",
    default=DEFAULT_HOST,
)
@click.option(
    "-p",
    "--port",
    show_default=True,
    type=int,
    help=f"WebApp Port. Default: {DEFAULT_GRADIO_PORT}",
    default=DEFAULT_GRADIO_PORT,
)
@click.option(
    "-c",
    "--rag-url",
    show_default=True,
    help=f"PAI-RAG service endpoint. Default: {DEFAULT_RAG_URL}",
    default=DEFAULT_RAG_URL,
)
def ui(host, port, rag_url):
    configure_webapp(app=app, web_url=f"http://{host}:{port}/", rag_url=rag_url)
    uvicorn.run(app, host=host, port=port, loop="asyncio")


@main.command()
@click.option(
    "-h",
    "--host",
    show_default=True,
    help=f"Host IP. Default: {DEFAULT_HOST}",
    default=DEFAULT_HOST,
)
@click.option(
    "-p",
    "--port",
    show_default=True,
    type=int,
    help=f"Port. Default: {DEFAULT_PORT}",
    default=DEFAULT_PORT,
)
@click.option(
    "-c",
    "--config-file",
    show_default=True,
    help=f"Configuration file. Default: {DEFAULT_APPLICATION_CONFIG_FILE}",
    default=DEFAULT_APPLICATION_CONFIG_FILE,
)
@click.option(
    "-w",
    "--workers",
    show_default=True,
    help="Worker Number. Default: 1",
    type=int,
    default=1,
)
@click.option(
    "-e",
    "--enable-example",
    show_default=True,
    help="whether to load example data. Default:False ",
    required=False,
    type=bool,
    default=False,
)
@click.option(
    "-s",
    "--skip-download-models",
    show_default=True,
    help="whether to skip download models from modelscope",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
)
def serve(host, port, config_file, workers, enable_example, skip_download_models):
    rag_configuration = RagConfiguration.from_file(config_file)
    init_trace(rag_configuration.get_value().get("RAG.trace"))

    if not skip_download_models and DEFAULT_MODEL_DIR != EAS_DEFAULT_MODEL_DIR:
        logger.info("Start to download models.")
        ModelScopeDownloader().load_basic_models()
        ModelScopeDownloader().load_mineru_config()
        logger.info("Finished downloading models.")
    else:
        logger.info("Start to loading minerU config file.")
        ModelScopeDownloader().load_mineru_config()
        logger.info("Finished loading minerU config file.")

    os.environ["PAI_RAG_MODEL_DIR"] = DEFAULT_MODEL_DIR
    app = FastAPI(lifespan=lifespan)
    configure_app(app, rag_configuration)
    uvicorn.run(app=app, host=host, port=port, loop="asyncio", workers=workers)
