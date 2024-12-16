from pai_rag.tools.data_process.ops.base_op import OPERATORS
from pai_rag.tools.data_process.ops.parser_op import Parser
from pai_rag.tools.data_process.utils.mm_utils import size_to_bytes
from pai_rag.tools.data_process.utils.cuda_utils import get_num_gpus, calculate_np


def load_ops(process_list):
    """
    Load op list according to the process list from config file.

    :param process_list: A process list. Each item is an op name and its
        arguments.
    :param op_fusion: whether to fuse ops that share the same intermediate
        variables.
    :return: The op instance list.
    """
    ops = []
    new_process_list = []
    op_names = []
    for process in process_list:
        op_name, args = list(process.items())[0]
        if op_name == "pai_rag_parser":
            if args.get("accelerator", "cpu") == "cuda":
                mem_required = (
                    size_to_bytes(args.get("mem_required", "1GB")) / 1024**3
                )
                cpu_required = args.get("cpu_required", 1)
                op_proc = calculate_np(op_name, mem_required, cpu_required, None, True)
                num_gpus = get_num_gpus(True, op_proc)
                RemoteGPUParser = Parser.options(
                    num_cpus=cpu_required, num_gpus=num_gpus
                )
                ops.append(RemoteGPUParser.remote(**args))
            else:
                num_cpus = args.get("cpu_required", 1)
                RemoteCPUParser = Parser.options(num_cpus=num_cpus)
                ops.append(RemoteCPUParser.remote(**args))
        else:
            ops.append(OPERATORS.modules[op_name](**args))
        new_process_list.append(process)
        op_names.append(op_name)

    for op_cfg, op in zip(new_process_list, ops):
        op._op_cfg = op_cfg

    return ops, op_names
