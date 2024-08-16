import logging
from typing import Dict, List, Any
from pai_rag.integrations.llms.dashscope.fc_base import MyFCDashScope
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class FunctionCallingLlmModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        llm_config = new_params[MODULE_PARAM_CONFIG]
        if llm_config and llm_config.source.lower() == "dashscope":
            model_name = llm_config.get("name", "qwen2-7b-instruct")
            logger.info(
                f"Using DashScope for Function-Calling-LLM with model: {model_name}."
            )
            return MyFCDashScope(model_name=model_name)
        elif llm_config and llm_config.source.lower() == "openai":
            model_name = llm_config.get("name", "gpt-3.5-turbo")
            logger.info(
                f"Using OpenAI for Function-Calling-LLM with model: {model_name}."
            )
            return MyFCDashScope(model_name=model_name)
        else:
            logger.info("Don't use Function-Calling-LLM.")
            return None
