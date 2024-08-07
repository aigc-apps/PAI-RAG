import logging
from typing import Dict, List, Any
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.tool.utils import (
    get_google_web_search_tools,
    get_calculator_tools,
    get_customized_tools,
    get_weather_tools,
)


logger = logging.getLogger(__name__)


class ToolModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.config = new_params[MODULE_PARAM_CONFIG]
        type = self.config["type"]
        logger.info(
            f"""
            [Parameters][Tool:FunctionTool]
                tools = {self.config.get("type", "Unknown tool")}
            """
        )
        tools = []
        if "googlewebsearch" in type:
            tools.extend(get_google_web_search_tools(self.config))

        if "calculator" in type:
            tools.extend(get_calculator_tools())

        if "custom" in type:
            tools.extend(get_customized_tools(self.config))

        if "weather" in type:
            tools.extend(get_weather_tools(self.config))

        return tools
