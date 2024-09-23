"""query engine factory based on config"""

import logging
from typing import Dict, List, Any
from pai_rag.integrations.query_transform.pai_query_transform import (
    PaiHyDEQueryTransform,
    PaiCondenseQueryTransform,
    PaiFusionQueryTransform,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG


logger = logging.getLogger(__name__)


class QueryTransformModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule", "ChatStoreModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        chat_store = new_params["ChatStoreModule"]

        if not config:
            logger.info("No Query Transform Configuration found")
            return None

        if config["type"] == "HyDEQueryTransform":
            logger.info("Query Transform: HyDEQueryTransform")
            return PaiHyDEQueryTransform(include_original=True)
        elif config["type"] == "FusionQueryTransform":
            logger.info("Query Transform: FusionQueryTransform")
            return PaiFusionQueryTransform()
        elif config["type"] == "CondenseQueryTransform":
            logger.info("Query Transform: CondenseQueryTransform")
            return PaiCondenseQueryTransform(chat_store=chat_store)
        else:
            return None
