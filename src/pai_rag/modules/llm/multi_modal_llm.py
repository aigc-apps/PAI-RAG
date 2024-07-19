import logging
from typing import Dict, List, Any
from llama_index.multi_modal_llms.dashscope import (
    DashScopeMultiModal,
    DashScopeMultiModalModels,
)

from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class MultiModalLlmModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        llm_config = new_params[MODULE_PARAM_CONFIG]
        if llm_config.source.lower() == "dashscope":
            logger.info("Using DashScope Multi-Modal-LLM.")
            return DashScopeMultiModal(model_name=DashScopeMultiModalModels.QWEN_VL_MAX)
        else:
            logger.info("Don't use Multi-Modal-LLM.")
            return None
