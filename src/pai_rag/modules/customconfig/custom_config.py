import logging
import json
import os
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
        agent_file_dir = config.get("agent_file_path", None)
        agent_config = None
        function_body_str = None
        if agent_file_dir:
            logger.info(
                f"""
                [Parameters][CustomConfig]
                    agent_file_path = {agent_file_dir}
                """
            )
            with open(
                os.path.join(agent_file_dir, "config.json"), "r", encoding="utf-8"
            ) as file:
                agent_config = json.load(file)
            if os.path.exists(os.path.join(agent_file_dir, "functions.py")):
                with open(
                    os.path.join(agent_file_dir, "functions.py"), "r", encoding="utf-8"
                ) as file:
                    function_body_str = file.read()
        else:
            logger.info("Don't use CustomConfig Module.")
        return agent_config, function_body_str
