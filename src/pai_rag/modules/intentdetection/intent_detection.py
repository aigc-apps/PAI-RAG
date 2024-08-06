import logging
from typing import Dict, List, Any
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.intentdetection.llm_single_detector import LLMSingleDetector
from llama_index.core.tools import ToolMetadata

logger = logging.getLogger(__name__)


class IntentDetectionModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        llm = new_params["LlmModule"]
        print("config", config)
        type = config.get("type", "single")
        intents = config.get("intent", None)
        if type == "single":
            logger.info(
                f"""
                [Parameters][IntentDetection]
                    type = {config.get("type", "Unknown inten detection type")}
                """
            )
            intents_tools = []
            for intent in intents:
                tool = ToolMetadata(description=intent[1], name=intent[0])
                intents_tools.append(tool)
            intent_detector = LLMSingleDetector(llm=llm, choices=intents_tools)
        else:
            raise ValueError(f"Unknown inten detection type: '{config['type']}'")

        return intent_detector
