import os
os.environ["ENABLE_FASTAPI"] = "false"
os.environ["ENABLE_REQUESTS"] = "false"
os.environ["ENABLE_AIOHTTPCLIENT"] = "false"
from aliyun.opentelemetry.instrumentation.auto_instrumentation import sitecustomize # ByPass unused import: use for arms trace initialization

import click
import uvicorn
from fastapi import FastAPI
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pathlib import Path

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


app = FastAPI()


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("-V", "--version", is_flag=True, help="Show version and exit.")
def run(ctx, version):
    if version:
        click.echo(version)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@run.command()
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
    from pai_rag.app.web.webui import configure_webapp

    configure_webapp(app=app, web_url=f"http://{host}:{port}/", rag_url=rag_url)
    uvicorn.run(app, host=host, port=port, loop="asyncio")


@run.command()
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
    from pai_rag.app.api.service import configure_app
    from pai_rag.utils.download_models import ModelScopeDownloader

    rag_configuration = RagConfigManager.from_file(config_file)
    rag_configuration.persist()
    ModelScopeDownloader().load_rag_models(skip_download_models)

    os.environ["PAI_RAG_MODEL_DIR"] = DEFAULT_MODEL_DIR
    app = FastAPI()
    configure_app(app, rag_configuration)
    uvicorn.run(app=app, host=host, port=port, loop="asyncio", workers=workers)


if __name__ == "__main__":
    serve()
