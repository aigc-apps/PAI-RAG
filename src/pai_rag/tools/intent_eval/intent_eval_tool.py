import click
import os
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.chat_flow import ChatFlow
from pai_rag.app.api.models import ChatCompletionRequest
from llama_index.core.base.llms.types import ChatMessage
import asyncio
import json

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
    "-f",
    "--test_file",
    type=str,
    required=True,
    help="file",
)
def run(
    config_file=None,
    test_file=None,
):
    config = RagConfigManager.from_file(config_file).get_value()
    chat_flow = ChatFlow()
    results = []
    scores = 0
    with open(test_file, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
            request_settings = data["request_settings"]
            samples = data["samples"]
            for sample in samples:
                chat_request = ChatCompletionRequest(
                    model=request_settings["model"],
                    messages=[ChatMessage(role="user", content=sample["query"])],
                    stream=request_settings["stream"],
                    chat_news=request_settings["chat_news"],
                    search_web=request_settings["search_web"],
                    temperature=request_settings["temperature"],
                )
                query_bundle = asyncio.run(
                    chat_flow._recognize_intent(chat_request, config)
                )
                sample["predicted_intent"] = query_bundle.intent
                sample["score"] = int(query_bundle.intent == sample["intent"])
                scores += sample["score"]
                results.append(sample)
            write_file_path = test_file.replace(".json", "_predicted.json")
            average_score = scores / len(samples)
            with open(write_file_path, "w", encoding="utf-8") as wfile:
                json.dump(
                    {
                        "request_settings": request_settings,
                        "samples": results,
                        "average_score": average_score,
                    },
                    wfile,
                    ensure_ascii=False,
                    indent=4,
                )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")


if __name__ == "__main__":
    run()
