from typing import Any, Dict, List
from pai_rag.utils.oss_client import OssClient
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
import logging
import os

logger = logging.getLogger(__name__)


class OssCacheModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        cache_config = new_params[MODULE_PARAM_CONFIG]
        use_oss = cache_config.get("enable", False)
        oss_ak = os.getenv("OSS_ACCESS_KEY_ID", None)
        oss_sk = os.getenv("OSS_ACCESS_KEY_SECRET", None)
        oss_bucket = cache_config.get("bucket", None)
        oss_endpoint = cache_config.get("endpoint", None)
        oss_prefix = cache_config.get("prefix", None)

        if use_oss:
            if oss_ak and oss_sk and oss_bucket and oss_endpoint:
                logger.info(f"Using OSS bucket {oss_bucket} for caching objects.")
                return OssClient(
                    bucket_name=oss_bucket, endpoint=oss_endpoint, prefix=oss_prefix
                )
            else:
                logger.warning(
                    "OSS config is incomplete. Will not cache objects. Please provide OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET, OSS_ENDPOINT."
                )
        else:
            logger.info("No OSS config provided. Will not cache objects.")
            return None
