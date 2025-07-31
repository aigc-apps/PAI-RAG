from loguru import logger
import torch
import os


USE_CUDA = os.environ.get("USE_CUDA", "false")


def should_use_cuda():
    if not torch.cuda.is_available():
        return False

    if USE_CUDA.lower() == "true" or USE_CUDA == "1":
        return True
    else:
        return False


def infer_cuda_device() -> str:
    if should_use_cuda():
        logger.info("Using cuda device.")
        return "cuda"
    else:
        logger.info(
            "Will not use CUDA device acceleration. If you want to use cuda, please set the environment variable USE_CUDA=1."
        )
        return "cpu"
