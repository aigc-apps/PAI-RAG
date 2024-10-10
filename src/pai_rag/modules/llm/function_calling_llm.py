import logging
from typing import Dict, List, Any
from pai_rag.integrations.llms.pai.llm_config import parse_llm_config
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class FunctionCallingLlmModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        if not config or not config.source:
            logger.info("Don't use Function-Calling-LLM.")
            return None

        llm_config = parse_llm_config(config)
        llm = PaiLlm(llm_config)
        logger.info(f"Function-calling LLM created: {llm.metadata}.")

        return llm
