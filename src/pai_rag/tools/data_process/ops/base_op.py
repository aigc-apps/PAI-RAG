from pai_rag.tools.data_process.utils.registry import Registry
from pai_rag.tools.data_process.utils.mm_utils import size_to_bytes
from pai_rag.tools.data_process.utils.cuda_utils import is_cuda_available
from loguru import logger

OPERATORS = Registry("Operators")


class BaseOP:
    _accelerator = "cpu"
    _batched_op = True

    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.get("batch_size", 10)
        _accelerator = kwargs.get("accelerator", None)
        if _accelerator is not None:
            self.accelerator = _accelerator
        else:
            self.accelerator = self._accelerator
        # parameters to determined the number of procs for this op
        self.num_proc = kwargs.get("num_proc", None)
        self.cpu_required = kwargs.get("cpu_required", 1)
        self.mem_required = kwargs.get("mem_required", 0)
        if isinstance(self.mem_required, str):
            self.mem_required = size_to_bytes(self.mem_required) / 1024**3
        logger.info(
            f"""Init OP with the following parameters:
                        Working dir: {kwargs.get("working_dir", None)}
                        Accelerator: {self.accelerator}
                        Num proc: {self.num_proc}
                        CPU required: {self.cpu_required}
                        Mem required: {self.mem_required}"""
        )

    @classmethod
    def is_batched_op(cls):
        return cls._batched_op

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def use_cuda(self):
        return self.accelerator == "cuda" and is_cuda_available()
