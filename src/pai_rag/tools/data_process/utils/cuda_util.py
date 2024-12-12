import os
import subprocess
import sys
from loguru import logger

# allow loading truncated images for some too large images.
from PIL import ImageFile
import importlib.metadata
import importlib.util
from typing import Tuple, Union

ImageFile.LOAD_TRUNCATED_IMAGES = True

# For now, only INFO will be shown. Later the severity level will be changed
# when setup_logger is called to initialize the logger.
logger.remove()
logger.add(sys.stderr, level="INFO")


def _is_package_available(
    pkg_name: str, return_version: bool = False
) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere
    # but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logger.debug(f"Detected {pkg_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def _cuda_device_count():
    _torch_available = _is_package_available("torch")

    if _torch_available:
        import torch

        return torch.cuda.device_count()

    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        all_devices = nvidia_smi_output.strip().split("\n")

        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES is ignored when torch is unavailable. "
                "All detected GPUs will be used."
            )

        return len(all_devices)
    except Exception:
        # nvidia-smi not found or other error
        return 0


_CUDA_DEVICE_COUNT = _cuda_device_count()


def cuda_device_count():
    return _CUDA_DEVICE_COUNT


def is_cuda_available():
    return _CUDA_DEVICE_COUNT > 0


def get_num_gpus(op, op_proc):
    if not op.use_cuda():
        return 0
    proc_per_gpu = op_proc / cuda_device_count()
    return 1.0 / proc_per_gpu
