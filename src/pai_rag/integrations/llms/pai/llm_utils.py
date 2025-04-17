import os
from typing import Sequence
from urllib.parse import urljoin
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from pai_rag.integrations.llms.pai.llm_config import (
    OpenAICompatibleLlmConfig,
    PaiBaseLlmConfig,
    OpenAILlmConfig,
    DashScopeLlmConfig,
    PaiEasLlmConfig,
    DEFAULT_MLLM_MAX_TOKENS,
)
from pai_rag.integrations.llms.pai.open_ai_alike_multi_modal import (
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


def create_llm(llm_config: PaiBaseLlmConfig):
    if isinstance(llm_config, OpenAILlmConfig):
        logger.info(
            f"""
            [Parameters][LLM:OpenAI]
                model = {llm_config.model},
                temperature = {llm_config.temperature},
                system_prompt = {llm_config.system_prompt}
            """
        )
        llm = OpenAI(
            model=llm_config.model,
            temperature=llm_config.temperature,
            system_prompt=llm_config.system_prompt,
            api_key=llm_config.api_key,
            max_tokens=llm_config.max_tokens,
            reuse_client=False,
        )
    elif isinstance(llm_config, DashScopeLlmConfig):
        logger.info(
            f"""
            [Parameters][LLM:DashScope]
                model = {llm_config.model},
                temperature = {llm_config.temperature},
                system_prompt = {llm_config.system_prompt}
            """
        )
        llm = OpenAILike(
            model=llm_config.model,
            api_base=llm_config.base_url,
            temperature=llm_config.temperature,
            system_prompt=llm_config.system_prompt,
            is_chat_model=True,
            api_key=llm_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
            max_tokens=llm_config.max_tokens,
            reuse_client=False,
            timeout=120,
        )
    elif isinstance(llm_config, PaiEasLlmConfig):
        logger.info(
            f"""
            [Parameters][LLM:PAI-EAS]
                model = {llm_config.model},
                endpoint = {llm_config.endpoint},
                token = {llm_config.token}
            """
        )
        llm = OpenAILike(
            model=llm_config.model,
            api_base=_make_openai_compatible_base_url(llm_config.endpoint),
            temperature=llm_config.temperature,
            system_prompt=llm_config.system_prompt,
            api_key=llm_config.token,
            is_chat_model=True,
            max_tokens=llm_config.max_tokens,
            reuse_client=False,
            timeout=120,
        )
    elif isinstance(llm_config, OpenAICompatibleLlmConfig):
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
            max_tokens=llm_config.max_tokens,
            reuse_client=False,
            timeout=120,
        )
    else:
        raise ValueError(f"Unknown LLM source: '{llm_config}'")

    return llm


def create_multi_modal_llm(llm_config: PaiBaseLlmConfig):
    max_tokens = min(llm_config.max_tokens, DEFAULT_MLLM_MAX_TOKENS)
    if isinstance(llm_config, OpenAILlmConfig):
        logger.info(
            f"""
            [Parameters][LLM:OpenAI]
                model = {llm_config.model},
                temperature = {llm_config.temperature},
                system_prompt = {llm_config.system_prompt}
            """
        )
        llm = OpenAIMultiModal(
            model=llm_config.model,
            temperature=llm_config.temperature,
            system_prompt=llm_config.system_prompt,
            api_key=llm_config.api_key,
            max_new_tokens=max_tokens,
        )
    elif isinstance(llm_config, DashScopeLlmConfig):
        logger.info(
            f"""
            [Parameters][LLM:DashScope]
                model = {llm_config.model},
                temperature = {llm_config.temperature},
                system_prompt = {llm_config.system_prompt}
            """
        )
        llm = OpenAIAlikeMultiModal(
            model=llm_config.model,
            api_base=llm_config.base_url,
            temperature=llm_config.temperature,
            system_prompt=llm_config.system_prompt,
            is_chat_model=True,
            api_key=llm_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
            max_new_tokens=max_tokens,
        )
    elif isinstance(llm_config, PaiEasLlmConfig):
        logger.info(
            f"""
            [Parameters][LLM:PAI-EAS]
                model = {llm_config.model},
                endpoint = {llm_config.endpoint},
            """
        )
        llm = OpenAIAlikeMultiModal(
            model=llm_config.model,
            api_base=llm_config.endpoint,
            temperature=llm_config.temperature,
            system_prompt=llm_config.system_prompt,
            api_key=llm_config.token,
            is_chat_model=True,
            max_new_tokens=max_tokens,
        )
    elif isinstance(llm_config, OpenAICompatibleLlmConfig):
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
            max_new_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown Multi-modal LLM source: '{llm_config}'")

    return llm


def merge_consecutive_messages(
    messages: Sequence[ChatMessage],
) -> Sequence[ChatMessage]:
    merged_messages = []
    if not messages:
        return merged_messages

    current_role = messages[0].role
    current_text = ""
    current_additional_kwargs = {}

    for message in messages:
        if message.role == current_role:
            current_text += message.content or ""
        else:
            merged_messages.append(
                ChatMessage(
                    role=current_role,
                    content=current_text,
                    additional_kwargs=current_additional_kwargs,
                )
            )
            current_role = message.role
            current_text = message.content or ""
            current_additional_kwargs = message.additional_kwargs

    merged_messages.append(
        ChatMessage(
            role=current_role,
            content=current_text,
            additional_kwargs=current_additional_kwargs,
        )
    )

    return merged_messages
