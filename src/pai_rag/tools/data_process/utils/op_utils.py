from loguru import logger
from pai_rag.tools.data_process.ops.base_op import OPERATORS
from pai_rag.tools.data_process.utils.mm_utils import size_to_bytes
from pai_rag.tools.data_process.utils.cuda_utils import get_num_gpus, calculate_np
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

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
                pg = placement_group(
                    [{"CPU": num_cpus, "GPU": num_gpus}] * int(op_proc),
                    strategy="SPREAD",
                )
                ray.get(pg.ready())
                logger.info(
                    f"Op {op_name} will be executed on cuda env and use {num_cpus} cpus and {num_gpus} GPUs."
                )
                RemoteGpuOp = OPERATORS.modules[op_name].options(
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg
                    ),
                )
                return RemoteGpuOp.remote(**op_args), pg
            else:
                logger.info(
                    f"Op {op_name} will be executed on cpu env and use {num_cpus} cpus."
                )
                op_proc = calculate_np(op_name, mem_required, num_cpus, None, False)
                pg = placement_group(
                    [{"CPU": num_cpus}] * int(op_proc), strategy="SPREAD"
                )
                ray.get(pg.ready())
                RemoteCpuOp = OPERATORS.modules[op_name].options(
                    num_cpus=num_cpus,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg
                    ),
                )
                return RemoteCpuOp.remote(**op_args), pg
        else:
            continue


def load_op_names(process_list):
    op_names = []
    for process in process_list:
        op_name, _ = list(process.items())[0]
        op_names.append(op_name)
    return op_names
