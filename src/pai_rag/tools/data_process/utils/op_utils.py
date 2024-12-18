from loguru import logger
from pai_rag.tools.data_process.ops.base_op import OPERATORS
from pai_rag.tools.data_process.utils.mm_utils import size_to_bytes
from pai_rag.tools.data_process.utils.cuda_utils import get_num_gpus, calculate_np

OPERATIONS = ["pai_rag_parser", "pai_rag_splitter", "pai_rag_embedder"]


def get_previous_operation(operation):
    try:
        index = OPERATIONS.index(operation)
        if index > 0:
            return OPERATIONS[index - 1]
        else:
            return None
    except ValueError:
        return None


def load_op(op_name, process_list):
    for process in process_list:
        name, op_args = list(process.items())[0]
        if name == op_name:
            mem_required = size_to_bytes(op_args.get("mem_required", "1GB")) / 1024**3
            num_cpus = op_args.get("cpu_required", 1)
            if op_args.get("accelerator", "cpu") == "cuda":
                op_proc = calculate_np(op_name, mem_required, num_cpus, None, True)
                num_gpus = get_num_gpus(True, op_proc)
                logger.info(
                    f"Op {op_name} will be executed on cuda env with op_proc {op_proc} and use {num_cpus} cpus and {num_gpus} GPUs."
                )
                RemoteGpuOp = OPERATORS.modules[op_name].options(
                    num_cpus=num_cpus, num_gpus=num_gpus, max_concurrency=op_proc
                )
                return RemoteGpuOp.remote(**op_args)
            else:
                op_proc = calculate_np(op_name, mem_required, num_cpus, None, False)
                logger.info(
                    f"Op {op_name} will be executed on cpu env with op_proc {op_proc} and use {num_cpus} cpus."
                )
                RemoteCpuOp = OPERATORS.modules[op_name].options(
                    num_cpus=num_cpus, max_concurrency=op_proc
                )
                return RemoteCpuOp.remote(**op_args)
        else:
            continue


def load_op_names(process_list):
    op_names = []
    for process in process_list:
        op_name, _ = list(process.items())[0]
        op_names.append(op_name)
    return op_names
