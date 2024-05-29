import click
import uvicorn
from pai_rag.app.app import create_app
from logging.config import dictConfig
import os
from pathlib import Path

_BASE_DIR = Path(__file__).parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


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
    "--config",
    show_default=True,
    help=f"Configuration file. Default: {DEFAULT_APPLICATION_CONFIG_FILE}",
    default=DEFAULT_APPLICATION_CONFIG_FILE,
)
def run(host, port, config):
    init_log()

    endpoint = f"http://{host}:{port}/"
    app = create_app(config, endpoint)
    uvicorn.run(app=app, host=host, port=port, loop="asyncio")
