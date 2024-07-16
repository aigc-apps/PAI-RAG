from typing import Any, Dict, List
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.integrations.nodes.raptor_nodes_enhance import RaptorNodesEnhancement
import logging

logger = logging.getLogger(__name__)


class NodesEnhancementModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]

        return RaptorNodesEnhancement(
            tree_depth=config.get("tree_depth", 3),
            max_clusters=config.get("max_clusters", 50),
            threshold=config.get("proba_threshold", 0.1),
        )
