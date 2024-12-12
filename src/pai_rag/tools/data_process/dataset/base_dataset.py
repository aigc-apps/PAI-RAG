from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base dataset of PaiRag DataProcess"""

    @abstractmethod
    def process(
        self,
        operators,  # TODO: add type hint
        *,
        exporter=None,
        checkpointer=None,
        tracer=None
    ):
        """process a list of operators on the dataset."""
        pass
