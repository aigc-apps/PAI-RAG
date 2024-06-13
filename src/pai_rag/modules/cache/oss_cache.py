from typing import Any, Dict, List
from pai_rag.utils.oss_cache import OssCache
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
import logging

logger = logging.getLogger(__name__)


class OssCacheModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        cache_config = new_params[MODULE_PARAM_CONFIG]
        if cache_config:
            return OssCache(cache_config)
        else:
            return None
