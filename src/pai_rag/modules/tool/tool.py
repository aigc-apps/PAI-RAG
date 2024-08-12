import logging
from typing import Dict, List, Any
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.tool.utils import (
    get_google_web_search_tools,
    get_calculator_tools,
    get_customized_python_tools_1,
    get_weather_tools,
    get_customized_api_tools,
)


logger = logging.getLogger(__name__)


class ToolModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["CustomConfigModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.config = new_params[MODULE_PARAM_CONFIG]
        agent_config, function_body_str = new_params["CustomConfigModule"]
        type = self.config["type"]
        logger.info(
            f"""
            [Parameters][Tool:FunctionTool]
                tools = {self.config.get("type", "Unknown tool")}
            """
        )
        tools = []
        if type == "built-in":
            name = self.config["name"]
            if "googlewebsearch" in name:
                tools.extend(get_google_web_search_tools(self.config))
            if "calculator" in name:
                tools.extend(get_calculator_tools())
            if "weather" in name:
                tools.extend(get_weather_tools(self.config))
        elif type == "python":
            tools.extend(
                get_customized_python_tools_1(agent_config["agent"], function_body_str)
            )
        elif type == "api":
            tools.extend(get_customized_api_tools(agent_config["agent"]))

        return tools
