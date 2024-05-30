import logging
from typing import Dict, List, Any
from llama_index.core.agent.react import ReActAgent
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class AgentModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule", "ToolModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        llm = new_params["LlmModule"]
        func_tool = new_params["ToolModule"]
        type = config["type"].lower()
        if type == "react":
            logger.info(
                f"""
                [Parameters][Agent]
                    type = {config.get("type", "Unknown agent type")}
                """
            )
            agent = ReActAgent.from_tools(tools=func_tool, llm=llm, verbose=True)
        else:
            raise ValueError(f"Unknown Agent type: '{config['type']}'")

        return agent
