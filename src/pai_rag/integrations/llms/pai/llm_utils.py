import os
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
)
from pai_rag.integrations.llms.pai.open_ai_alike_multi_modal import (
    OpenAIAlikeMultiModal,
)

from loguru import logger


def _make_openai_compatible_base_url(base_url: str):
    if base_url.endswith("/v1"):
        return base_url

    return urljoin(base_url.rstrip("/") + "/", "v1")


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
            max_new_tokens=llm_config.max_tokens,
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
            max_new_tokens=llm_config.max_tokens,
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
        llm = OpenAIAlikeMultiModal(
            model=llm_config.model,
            api_base=llm_config.endpoint,
            temperature=llm_config.temperature,
            system_prompt=llm_config.system_prompt,
            api_key=llm_config.token,
            is_chat_model=True,
            max_new_tokens=llm_config.max_tokens,
        )
    elif isinstance(llm_config, OpenAICompatibleLlmConfig):
        api_base = _make_openai_compatible_base_url(llm_config.base_url)
        logger.info(
            f"""
            [Parameters][LLM:OpenAICompatible]
                model = {llm_config.model},
                base_url = {api_base},
                endpoint = {llm_config.base_url},
                token = {llm_config.api_key}
            """
        )
        llm = OpenAIAlikeMultiModal(
            model=llm_config.model,
            api_base=api_base,
            temperature=llm_config.temperature,
            system_prompt=llm_config.system_prompt,
            api_key=llm_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
            is_chat_model=True,
            max_new_tokens=llm_config.max_tokens,
        )
    else:
        raise ValueError(f"Unknown Multi-modal LLM source: '{llm_config}'")

    return llm
