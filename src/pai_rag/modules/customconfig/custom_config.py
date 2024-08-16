import logging
import json
from typing import Dict, List, Any
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class CustomConfigModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        agent_file_path = config.get("agent_file_path", None)
        function_file_path = config.get("function_file_path", None)
        agent_config = None
        function_body_str = None
        if agent_file_path:
            logger.info(
                f"""
                [Parameters][CustomConfig]
                    agent_file_path = {agent_file_path}
                """
            )
            with open(agent_file_path, "r", encoding="utf-8") as file:
                agent_config = json.load(file)
        if function_file_path:
            with open(function_file_path, "r", encoding="utf-8") as file:
                function_body_str = file.read()
        else:
            logger.info("Don't use CustomConfig Module.")
        return agent_config, function_body_str
