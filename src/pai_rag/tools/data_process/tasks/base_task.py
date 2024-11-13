from pai_rag.tools.data_process.utils.cuda_util import is_cuda_available
from pai_rag.core.rag_config_manager import RagConfigManager


class BaseTask:
    """A base class for tasks that process a dataset."""

    _accelerator = "cpu"

    def __init__(self, **kwargs):
        """
        Initializes the BaseTask instance.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments. It can contain:
                - accelerator: a string indicating the computational accelerator ('cpu' or 'cuda').
        """
        self.accelerator = kwargs.get("accelerator", self._accelerator)
        self.config_file = kwargs.get("config_file", None)
        if self.config_file is not None:
            self.config = RagConfigManager.from_file(self.config_file).get_value()
        else:
            raise ValueError("Config file is not provided")
        self.oss_path = kwargs.get("oss_path", None)
        self.data_path = kwargs.get("data_path", None)

    @classmethod
    def is_batched_task(cls):
        return cls._batched_task

    def use_cuda(self):
        return self.accelerator == "cuda" and is_cuda_available()

    def process(self):
        raise NotImplementedError
