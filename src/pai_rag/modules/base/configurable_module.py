from abc import ABC, abstractmethod
from typing import Dict, List, Any

DEFAULT_INSTANCE_KEY = "__DEFAULT_INSTANCE__"


class ConfigurableModule(ABC):
    """Configurable Module

    Helps to create instances according to configuration.
    """

    def __init__(self):
        self.__params_map = {}
        self.__instance_map = {}

    @abstractmethod
    def _create_new_instance(self, new_params: Dict[str, Any]):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_dependencies() -> List[str]:
        raise NotImplementedError

    def get_or_create(self, new_params: Dict[str, Any]):
        return self.get_or_create_by_name(new_params=new_params)

    def get_or_create_by_name(
        self, new_params: Dict[str, Any], name: str = DEFAULT_INSTANCE_KEY
    ):
        # Create new instance when initializing or config changed.
        if (
            self.__params_map.get(name, None) is None
            or self.__params_map[name] != new_params
        ):
            print(f"{self.__class__.__name__} param changed, updating")
            self.__instance_map[name] = self._create_new_instance(new_params)
            self.__params_map[name] = new_params
        else:
            print(f"{self.__class__.__name__} param unchanged, skipping")

        return self.__instance_map[name]
