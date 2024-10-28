import click
import os
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_module import resolve_agent
import logging

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")


@click.command()
@click.option(
    "-c",
    "--config_file",
    show_default=True,
    help=f"Configuration file. Default: {DEFAULT_APPLICATION_CONFIG_FILE}",
    default=DEFAULT_APPLICATION_CONFIG_FILE,
)
@click.option(
    "-q",
    "--question",
    type=str,
    required=True,
    help="question",
)
@click.option(
    "-s",
    "--stream",
    type=bool,
    default=False,
    required=False,
    is_flag=True,
    help="stream mode",
)
@click.option(
    "-t",
    "--tool_definition_file",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "-p",
    "--python_script_file",
    type=str,
    default=None,
    required=False,
)
def run(
    config_file=None,
    question=None,
    stream=False,
    tool_definition_file=None,
    python_script_file=None,
):
    logging.basicConfig(level=logging.DEBUG)

    config = RagConfigManager.from_file(config_file).get_value()
    if tool_definition_file:
        config.agent.tool_definition_file = tool_definition_file
    if python_script_file:
        config.agent.python_script_file = python_script_file

    agent = resolve_agent(config)

    print("**Question**: ", question)
    response = agent.chat(question)
    print("**Answer**: ", response.response)


if __name__ == "__main__":
    run()
