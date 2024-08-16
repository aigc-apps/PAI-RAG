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
        return ["FunctionCallingLlmModule", "CustomConfigModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        llm = new_params["FunctionCallingLlmModule"]
        agent_config, _ = new_params["CustomConfigModule"]
        type = config.get("type", "single")
        if type == "single":
            logger.info(
                f"""
                [Parameters][IntentDetection]
                    type = {config.get("type", "Unknown inten detection type")}
                """
            )
            intents_tools = []
            if agent_config:
                intents = []
                for name in agent_config["intent"]:
                    intents.append([name, agent_config["intent"][name]])
            else:
                intents = config.get("intent", None)

            for intent in intents:
                tool = ToolMetadata(name=intent[0], description=intent[1])
                intents_tools.append(tool)
            intent_detector = LLMSingleDetector(llm=llm, choices=intents_tools)
            return intent_detector
        else:
            logger.info("Don't use IntentDetection Module.")
            return None
