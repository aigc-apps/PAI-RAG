def _cuda_device_count():
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        # torch package not found or other error
        return 0


_CUDA_DEVICE_COUNT = _cuda_device_count()


def cuda_device_count():
    return _CUDA_DEVICE_COUNT


def is_cuda_available():
    return _CUDA_DEVICE_COUNT > 0
