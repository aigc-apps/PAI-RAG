from pairag.data_pipeline.utils.cuda_utils import is_cuda_available


OUTPUT_BATCH_SIZE = 50000


class BaseOperator:
    def __init__(
        self,
        name: str = "default",
        num_cpus: float = 1,
        num_gpus: float = 0,
        model_dir: str = None,
        **kwargs,
    ):
        self.name = name
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.model_dir = model_dir
        self.kwargs = kwargs

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def use_cuda(self):
        return self.num_gpus > 0 and is_cuda_available()
