import os
from typing import Sequence
from urllib.parse import urljoin
from llama_index.llms.openai_like import OpenAILike
from pairag.integrations.llms.pai.llm_config import (
    OpenAICompatibleLlmConfig,
)
from pairag.integrations.llms.pai.open_ai_alike_multi_modal import (
    OpenAIAlikeMultiModal,
)
from llama_index.core.base.llms.types import ChatMessage

from loguru import logger


def _should_url_add_v1(base_url: str):
    if base_url.endswith("/v1") or base_url.endswith("/v1/"):
        return False

    elif "pai-eas" in base_url or "localhost" in base_url or "127.0.0.1" in base_url:
        return True

    return False


def _make_openai_compatible_base_url(base_url: str):
    if _should_url_add_v1(base_url):
        return urljoin(base_url.rstrip("/") + "/", "v1")
    return base_url


def create_llm(llm_config: OpenAICompatibleLlmConfig):
    api_base = _make_openai_compatible_base_url(llm_config.base_url)
    logger.info(
        f"""
        [Parameters][LLM:OpenAICompatible]
            model = {llm_config.model},
            base_url = {api_base},
            temperature = {llm_config.temperature},
            system_prompt = {llm_config.system_prompt}
        """
    )
    llm = OpenAILike(
        model=llm_config.model,
        api_base=api_base,
        temperature=llm_config.temperature,
        system_prompt=llm_config.system_prompt,
        is_chat_model=True,
        api_key=llm_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
        context_window=llm_config.context_window,
        reuse_client=False,
        timeout=120,
    )

    return llm


def create_multi_modal_llm(llm_config: OpenAICompatibleLlmConfig):
    api_base = _make_openai_compatible_base_url(llm_config.base_url)
    logger.info(
        f"""
        [Parameters][LLM:OpenAICompatible]
            model = {llm_config.model},
            base_url = {api_base},
        """
    )
    llm = OpenAIAlikeMultiModal(
        model=llm_config.model,
        api_base=api_base,
        temperature=llm_config.temperature,
        system_prompt=llm_config.system_prompt,
        api_key=llm_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
        is_chat_model=True,
    )
    return llm


def merge_consecutive_messages(
    messages: Sequence[ChatMessage],
) -> Sequence[ChatMessage]:
    merged_messages = []
    if not messages:
        return merged_messages

    current_role = messages[0].role
    current_text = ""

    for message in messages:
        if message.role == current_role:
            current_text += message.content
        else:
            merged_messages.append(ChatMessage(role=current_role, content=current_text))
            current_role = message.role
            current_text = message.content

    merged_messages.append(ChatMessage(role=current_role, content=current_text))

    return merged_messages
