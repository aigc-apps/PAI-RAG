import os
import json
import pytest
from pathlib import Path
from llama_index.core.base.llms.types import ChatMessage
import asyncio


if "DASHSCOPE_API_KEY" not in os.environ:
    pytest.skip(
        allow_module_level=True,
        reason='Environment variable "DASHSCOPE_API_KEY" not set.',
    )

from core.rag_config_manager import RagConfigManager
from chat.chat_flow import ChatFlow
from chat.models import ChatCompletionRequest
from integrations.llms.pai.llm_config import OpenAICompatibleLlmConfig


# 定义测试文件路径
_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")
TEST_FILE = os.path.join(_BASE_DIR, "intentdetection/multi_intents_sample.json")


@pytest.fixture(scope="module")
def setup_config():
    # 加载配置文件
    config = RagConfigManager.from_file(DEFAULT_APPLICATION_CONFIG_FILE).get_value()
    # 设置 LLM 配置
    llm_config = OpenAICompatibleLlmConfig(
        model="qwen2.5-32b-instruct", api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    config.llms[0] = llm_config
    return config


@pytest.fixture(scope="module")
def test_data():
    # 读取测试文件
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def test_intent_detection(setup_config, test_data):
    chat_flow = ChatFlow(setup_config)
    request_settings = test_data["request_settings"]
    samples = test_data["samples"]
    scores = 0
    results = []

    for sample in samples:
        chat_request = ChatCompletionRequest(
            model=request_settings["model"],
            messages=[ChatMessage(role="user", content=sample["query"])],
            stream=request_settings["stream"],
            chat_knowledgebase=request_settings["chat_knowledgebase"],
            search_web=request_settings["search_web"],
            chat_db=request_settings["chat_db"],
            temperature=request_settings["temperature"],
        )
        intent_result = asyncio.run(
            chat_flow._recognize_intent(chat_request, chat_history_str="")
        )
        sample["intent_output"] = {}
        sample["intent_output"]["intent_name"] = intent_result.intent

        sample["score"] = int(
            sample["intent_output"]["intent_name"] == sample["intent"]["intent_name"]
        )
        scores += sample["score"]

        results.append(sample)

    average_score = scores / len(samples)
    write_file_path = TEST_FILE.replace(".json", "_predicted.json")
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

    assert average_score == 1, f"Average score is {average_score}, expected 1"
