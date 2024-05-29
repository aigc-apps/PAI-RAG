from llama_index.llms.paieas.base import PaiEAS
from llama_index.core.base.llms.types import ChatMessage
import os


def _get_eas_llm() -> PaiEAS:
    eas_url = os.environ.get("TEST_EAS_URL", None)
    eas_token = os.environ.get("TEST_EAS_TOKEN", None)

    if not eas_url or not eas_token:
        return None

    eas_llm = PaiEAS(endpoint=eas_url, token=eas_token, model_name="EasCustomModel")
    return eas_llm


def test_pai_eas_llm_complete():
    eas_llm = _get_eas_llm()
    if eas_llm:
        response = eas_llm.complete("你好呀，最近怎么样？")
        assert len(response.text) > 0


def test_pai_eas_llm_stream_complete():
    eas_llm = _get_eas_llm()

    if eas_llm:
        response = eas_llm.stream_complete("你好呀，最近怎么样？")
        text = None
        for r in response:
            text = r.text

        assert len(text) > 0


def test_pai_eas_llm_chat():
    eas_llm = _get_eas_llm()
    if eas_llm:
        chat_messages = [ChatMessage(role="user", content="你好呀，最近怎么样？")]
        response = eas_llm.chat(chat_messages)
        print(response.message.content)
        assert len(response.message.content) > 0


def test_pai_eas_llm_stream_chat():
    eas_llm = _get_eas_llm()
    if eas_llm:
        chat_messages = [ChatMessage(role="user", content="你好呀，最近怎么样？")]
        response = eas_llm.stream_chat(chat_messages)

        text = None
        for r in response:
            text = r.message.content
        print(text)
        assert len(text) > 0
