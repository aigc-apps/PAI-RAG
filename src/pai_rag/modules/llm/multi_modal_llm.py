import logging
import os
from typing import Dict, List, Any
from pai_rag.integrations.llms.multimodal.open_ai_alike_multi_modal import (
    OpenAIAlikeMultiModal,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_MAX_NEW_TOKENS = 1500
DEFAULT_EAS_MAX_NEW_TOKENS = 1500


class MultiModalLlmModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        llm_config = new_params[MODULE_PARAM_CONFIG]
        if llm_config is None:
            logger.info("Don't use Multi-Modal-LLM.")
            return None
        if llm_config.source.lower() == "dashscope":
            model_name = llm_config.get("name", "qwen-vl-max")
            logger.info(
                f"""
                            [Parameters][Multi-Modal-LLM:DashScope]
                                model = {model_name}
                            """
            )
            return OpenAIAlikeMultiModal(
                model=model_name,
                api_base=DEFAULT_DASHSCOPE_API_BASE,
                api_key=os.environ.get("DASHSCOPE_API_KEY"),
                max_new_tokens=DEFAULT_DASHSCOPE_MAX_NEW_TOKENS,
            )
        elif llm_config.source.lower() == "paieas" and llm_config.get("endpoint"):
            logger.info("Using PAI-EAS Multi-Modal-LLM.")
            return OpenAIAlikeMultiModal(
                model=llm_config.get(
                    "name", "/model_repository/MiniCPM-V-2_6"
                ),  # TODO: change model path
                api_base=llm_config.endpoint,
                api_key=llm_config.token,
                max_new_tokens=DEFAULT_EAS_MAX_NEW_TOKENS,
            )
        else:
            logger.info("Don't use Multi-Modal-LLM.")
            return None
