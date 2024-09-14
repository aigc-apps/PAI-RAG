from typing import Any, Dict, List
from pai_rag.integrations.readers.pai.pai_data_reader import (
    BaseDataReaderConfig,
    PaiDataReader,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
import logging

logger = logging.getLogger(__name__)


class DataReaderModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["OssCacheModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.oss_cache = new_params["OssCacheModule"]
        reader_config = BaseDataReaderConfig.model_validate(
            new_params[MODULE_PARAM_CONFIG]
        )
        return PaiDataReader(reader_config, self.oss_cache)
