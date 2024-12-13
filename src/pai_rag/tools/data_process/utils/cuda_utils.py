import os
import subprocess
import sys
from loguru import logger
import math
import psutil
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


def get_min_cuda_memory():
    # get cuda memory info using "nvidia-smi" command
    import torch

    min_cuda_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
    nvidia_smi_output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    ).decode("utf-8")
    for line in nvidia_smi_output.strip().split("\n"):
        free_memory = int(line)
        min_cuda_memory = min(min_cuda_memory, free_memory)
    return min_cuda_memory


def calculate_np(name, mem_required, cpu_required, num_proc=None, use_cuda=False):
    """Calculate the optimum number of processes for the given OP"""
    eps = 1e-9  # about 1 byte

    if num_proc is None:
        num_proc = psutil.cpu_count()

    if use_cuda:
        cuda_mem_available = get_min_cuda_memory() / 1024
        op_proc = min(
            num_proc,
            math.floor(cuda_mem_available / (mem_required + eps)) * cuda_device_count(),
        )
        if use_cuda and mem_required == 0:
            logger.warning(
                f"The required cuda memory of Op[{name}] "
                f"has not been specified. "
                f"Please specify the mem_required field in the "
                f"config file, or you might encounter CUDA "
                f"out of memory error. You can reference "
                f"the mem_required field in the "
                f"config_all.yaml file."
            )
        if op_proc < 1.0:
            logger.warning(
                f"The required cuda memory:{mem_required}GB might "
                f"be more than the available cuda memory:"
                f"{cuda_mem_available}GB."
                f"This Op[{name}] might "
                f"require more resource to run."
            )
        op_proc = max(op_proc, 1)
        return op_proc
    else:
        op_proc = num_proc
        cpu_available = psutil.cpu_count()
        mem_available = psutil.virtual_memory().available
        mem_available = mem_available / 1024**3
        op_proc = min(op_proc, math.floor(cpu_available / cpu_required + eps))
        op_proc = min(op_proc, math.floor(mem_available / (mem_required + eps)))
        if op_proc < 1.0:
            logger.warning(
                f"The required CPU number:{cpu_required} "
                f"and memory:{mem_required}GB might "
                f"be more than the available CPU:{cpu_available} "
                f"and memory :{mem_available}GB."
                f"This Op [{name}] might "
                f"require more resource to run."
            )
        op_proc = max(op_proc, 1)
        return op_proc
