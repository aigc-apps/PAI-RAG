import sys
import math
import psutil
from loguru import logger


def compute_concurrency_count(
    num_cpus: int,
    memory: int,
    num_gpus: int = 0,
):
    concurrency = sys.maxsize

    if num_gpus > 0:
        import torch

        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count > 0:
            concurrency = math.floor(min(concurrency, cuda_device_count // num_gpus))
            logger.info(
                f"Available CUDA devices: {cuda_device_count}, required gpus {num_gpus}, updated conccurency {concurrency}."
            )
        else:
            logger.error("No CUDA devices found.")
            raise ValueError("No CUDA devices found.")

    cpu_available = psutil.cpu_count() - 4
    mem_available = psutil.virtual_memory().available
    mem_available = mem_available / 1024**3

    concurrency = math.floor(min(concurrency, cpu_available // num_cpus))
    logger.info(
        f"Available CPUs: {cpu_available}, required cpus {num_cpus}, updated conccurency {concurrency}."
    )

    concurrency = math.floor(min(concurrency, mem_available // memory))
    logger.info(
        f"Available memory: {mem_available}GB, required memory {memory}GB, updated conccurency {concurrency}."
    )

    return concurrency
