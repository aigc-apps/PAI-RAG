"""synthesizer factory, used to generate synthesizer instance based on customer config"""

import logging
import os
from typing import Dict, List, Any

from pai_rag.integrations.search.bing_search import BingSearchTool
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class SearchModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["EmbeddingModule", "SynthesizerModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG] or {}
        api_key = config.get("search_api_key") or os.environ.get("BING_SEARCH_KEY")
        if not api_key:
            logger.info("[AiSearch] Not enabled.")
            return None

        embed_model = new_params["EmbeddingModule"]
        synthesizer = new_params["SynthesizerModule"]

        logger.info("[AiSearch] Using BING searcher.")
        return BingSearchTool(
            api_key=api_key,
            embed_model=embed_model,
            synthesizer=synthesizer,
            search_count=config.get("search_count"),
            search_lang=config.get("search_lang"),
        )
