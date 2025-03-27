from dynaconf import Dynaconf, loaders
from dynaconf.utils.boxing import DynaBox

from loguru import logger
import os

from pai_rag.core.rag_config import RagConfig
from pai_rag.utils.oss_utils import check_and_set_oss_auth
from pai_rag.integrations.llms.pai.llm_config import PaiBaseLlmConfig

# store config file generated from ui.
GENERATED_CONFIG_FILE_NAME = "localdata/settings.snapshot.toml"


class RagConfigManager:
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_snapshot(cls):
        try:
            settings_files = [GENERATED_CONFIG_FILE_NAME]
            config = Dynaconf(
                # don't respect env when checking snapshot
                envvar_prefix="SOME_DUMMY_PREFIX",
                settings_file=settings_files,
                merge=True,
            )
            return cls(config)
        except Exception as error:
            logger.critical("Read config file failed.")
            raise error

    @classmethod
    def from_file(cls, config_file):
        try:
            settings_files = [config_file]
            config = Dynaconf(
                envvar_prefix="PAIRAG",
                settings_file=settings_files,
                merge=True,
            )
            snapshot_config = Dynaconf(settings_file=[GENERATED_CONFIG_FILE_NAME])
            config.update(snapshot_config, tomlfy=True, merge=True)
            config["rag"]["embedding"]["source"] = "huggingface"
            config["rag"]["index"]["vector_store"]["persist_path"] = config["rag"][
                "index"
            ]["persist_path"]
            config["rag"]["index"]["vector_store"]["type"] = config["rag"]["index"][
                "vector_store"
            ]["type"].lower()
            if "api_key" in config["rag"]["llm"]:
                config["rag"]["llm"]["api_key"] = str(config["rag"]["llm"]["api_key"])

            return cls(config)
            # `envvar_prefix` = export envvars with `export PAIRAG_FOO=bar`.
            # `settings_files` = Load these files in the order.
        except Exception as error:
            logger.critical("Read config file failed.")
            raise error

    def get_value(self) -> RagConfig:
        self.config.rag["llms"] = [item for item in self.config.rag["llms"] if item]
        rag_config = RagConfig.model_validate(self.config.rag)
        rag_config_copy = rag_config
        # 兼容之前的配置
        if not rag_config_copy.llms or len(rag_config_copy.llms) == 0:
            updated_llm = rag_config.llm.copy(
                update={
                    "vision_support": False,
                    "model_id": "default",
                    "is_reasoning_models": False,
                }
            )
            rag_config.llms.append(updated_llm)
        if rag_config.multimodal_llm.is_validate() and (
            not rag_config_copy.llms or len(rag_config_copy.llms) == 0
        ):
            updated_vllm = rag_config.multimodal_llm.copy(
                update={"vision_support": True, "is_reasoning_models": False}
            )
            rag_config.llms.append(updated_vllm)
        return rag_config

    def update(self, new_value: Dynaconf):
        if self.config.get("rag", None):
            self.config.rag.update(new_value, merge=True)
            check_and_set_oss_auth(self.config.rag)

    def persist(self):
        """Save configuration to file."""
        data = self.config.as_dict()
        if data["RAG"]["llms"]:
            data["RAG"]["llms"] = [
                llm.model_dump()
                for llm in data["RAG"]["llms"]
                if isinstance(llm, PaiBaseLlmConfig)
            ]
        os.makedirs("localdata", exist_ok=True)
        loaders.write(GENERATED_CONFIG_FILE_NAME, DynaBox(data).to_dict())
        return self.get_config_mtime()

    def get_config_mtime(self):
        try:
            return os.path.getmtime(GENERATED_CONFIG_FILE_NAME)
        except Exception as ex:
            logger.critical(f"Fail to read config mtime {ex}")
            return -1
