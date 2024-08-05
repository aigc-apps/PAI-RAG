from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging

DEFAULT_INSTANCE_KEY = "__DEFAULT_INSTANCE__"


logger = logging.getLogger(__name__)


class ConfigurableModule(ABC):
    """Configurable Module

    Helps to create instances according to configuration.
    """

    @abstractmethod
    def _create_new_instance(self, new_params: Dict[str, Any]):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_dependencies() -> List[str]:
        raise NotImplementedError

    def get_or_create(self, new_params: Dict[str, Any]):
        return self._create_new_instance(new_params)
