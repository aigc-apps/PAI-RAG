import asyncio
from contextlib import asynccontextmanager
import click
import uvicorn
from fastapi import FastAPI
from pai_rag.app.api.service import configure_app
from pai_rag.app.web.webui import configure_webapp
from logging.config import dictConfig
import os
from pathlib import Path

_BASE_DIR = Path(__file__).parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")
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
def serve(host, port, config_file, workers):
    app = FastAPI(lifespan=lifespan)
    configure_app(app, config_file=config_file)
    uvicorn.run(app=app, host=host, port=port, loop="asyncio", workers=workers)
