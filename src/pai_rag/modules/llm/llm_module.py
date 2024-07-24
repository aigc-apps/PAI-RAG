import logging
from typing import Dict, List, Any
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI

import os

# from llama_index.llms.dashscope import DashScope
from pai_rag.integrations.llms.dashscope.base import MyDashScope
from pai_rag.integrations.llms.dashscope.fc_base import MyFCDashScope
from pai_rag.integrations.llms.paieas.base import PaiEAS
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class LlmModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        source = config["source"].lower()

        # Get DASHSCOPE KEY, will use dashscope LLM
        # Since we might have already configured key for dashscope mebedding and multi-modal
        # Will refine later.
        if os.getenv("DASHSCOPE_API_KEY", None):
            model_name = config.get("name", "qwen-turbo")
            if config.get("type", None) == "function_calling":
                logger.info(
                    f"""
                    [Parameters][LLM:DashScope-FunctionCalling]
                        model = {model_name}
                    """
                )
                llm = MyFCDashScope(model_name=model_name)
            else:
                logger.info(
                    f"""
                    [Parameters][LLM:DashScope]
                        model = {model_name}
                    """
                )
                llm = MyDashScope(
                    model_name=model_name,
                    temperature=config.get("temperature", 0.1),
                    max_tokens=2000,
                )
        elif source == "openai":
            logger.info(
                f"""
                [Parameters][LLM:OpenAI]
                    model = {config.get("name", "gpt-3.5-turbo")},
                    temperature = {config.get("temperature", 0.1)},
                    system_prompt = {config.get("system_prompt", "Please answer in Chinese.")}
                """
            )
            llm = OpenAI(
                model=config.get("name", "gpt-3.5-turbo"),
                temperature=config.get("temperature", 0.1),
                system_prompt=config.get("system_prompt", "Please answer in Chinese."),
                api_key=config.get("api_key", None),
            )
        elif source == "azureopenai":
            logger.info(
                f"""
                [Parameters][LLM:AzureOpenAI]
                    model = {config.get("name", "gpt-35-turbo")},
                    temperature = {config.get("temperature", 0.1)},
                    system_prompt = {config.get("system_prompt", "Please answer in Chinese.")}
                """
            )
            llm = AzureOpenAI(
                model=config.get("name", "gpt-35-turbo"),
                temperature=config.get("temperature", 0.1),
                system_prompt=config.get("system_prompt", "Please answer in Chinese."),
            )
        elif source == "dashscope":
            model_name = config.get("name", "qwen-turbo")
            logger.info(
                f"""
                [Parameters][LLM:DashScope]
                    model = {model_name}
                """
            )
            llm = MyDashScope(
                model_name=model_name,
                temperature=config.get("temperature", 0.1),
                max_tokens=2000,
            )
        elif source == "paieas":
            model_name = config.get("name", "PAI-EAS-LLM")
            endpoint = config["endpoint"]
            token = config["token"]
            logger.info(
                f"""
                [Parameters][LLM:PAI-EAS]
                    model = {model_name},
                    endpoint = {endpoint},
                    token = {token}
                """
            )
            from urllib.parse import urljoin

            llm = PaiEAS(api_key=token, api_base=urljoin(endpoint, "v1"))
        else:
            raise ValueError(f"Unknown LLM source: '{config['llm']['source']}'")

        Settings.llm = llm
        return llm
