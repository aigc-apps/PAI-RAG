"""
def calculate_concurrency_count(
    name, mem_required, cpu_required, num_gpus=0, use_cuda=False
):
    eps = 1e-9  # about 1 byte

    if use_cuda:
        total_gpus = cuda_device_count()
        concurrency = total_gpus / num_gpus
        concurrency = math.floor(concurrency * 100) / 100
        if use_cuda and num_gpus == 0:
            logger.warning(
                "The required num_gpus"
                "has not been specified. "
                "Please specify the num_gpus field in the "
                "arguments or config file, or you might encounter CUDA "
                "out of memory error."
            )
        if concurrency < 1.0:
            logger.warning(
                f"Resource insufficient: he required gpu num {num_gpus} might "
                f"be more than total GPU device count {total_gpus}."
            )
        concurrency = max(concurrency, 1)
        return concurrency
    else:
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
"""
