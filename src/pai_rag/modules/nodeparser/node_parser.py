from typing import Any, Dict, List
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import (
    NodeParserConfig,
    PaiNodeParser,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG


class NodeParserModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["EmbeddingModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.embed_model = new_params["EmbeddingModule"]
        parser_config = NodeParserConfig.model_validate(new_params[MODULE_PARAM_CONFIG])
        return PaiNodeParser(parser_config)
