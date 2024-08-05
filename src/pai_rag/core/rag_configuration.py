from dynaconf import Dynaconf, loaders
from dynaconf.utils.boxing import DynaBox

import logging
import os

# store config file generated from ui.
GENERATED_CONFIG_FILE_NAME = "localdata/settings.snapshot.toml"


class RagConfiguration:
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
            logging.critical("Read config file failed.")
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
            return cls(config)
            # `envvar_prefix` = export envvars with `export PAIRAG_FOO=bar`.
            # `settings_files` = Load these files in the order.
        except Exception as error:
            logging.critical("Read config file failed.")
            raise error

    def get_value(self, key=None):
        key = key or "rag"  # use rag key as default config
        return self.config[key]

    def update(self, new_value: Dynaconf):
        if self.config.get("rag", None):
            self.config.rag.update(new_value, tomlfy=True, merge=True)

    def persist(self):
        """Save configuration to file."""
        data = self.config.as_dict()
        os.makedirs("localdata", exist_ok=True)
        loaders.write(GENERATED_CONFIG_FILE_NAME, DynaBox(data).to_dict(), merge=True)

    def get_config_mtime(self):
        try:
            return os.path.getmtime(GENERATED_CONFIG_FILE_NAME)
        except Exception as ex:
            print(f"Fail to read config mtime {ex}")
            return -1
