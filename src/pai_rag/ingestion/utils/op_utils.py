import os
from loguru import logger
from pathlib import Path
from pai_rag.ingestion.ops.base_op import OPERATORS
from pai_rag.ingestion.utils.mm_utils import size_to_bytes
from pai_rag.ingestion.utils.cuda_utils import get_num_gpus, calculate_np

OPERATIONS = ["rag_parser", "rag_splitter", "rag_embedder"]


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
                if op_args.get("concurrency", 0) > 0:
                    op_proc_num = op_args.get("concurrency", 0)
                else:
                    op_proc_num = calculate_np(
                        op_name, mem_required, num_cpus, None, True
                    )

                if op_args.get("node_concurrency", 0) > 0:
                    node_concurrency = op_args.get("node_concurrency")
                else:
                    node_concurrency = op_proc_num

                num_gpus = get_num_gpus(True, node_concurrency)
                logger.info(
                    f"Op {op_name} will be executed on cuda env with op_proc_num {op_proc_num} and use {num_cpus} cpus and {num_gpus} GPUs."
                )
                # 并发actor
                return [
                    OPERATORS.modules[op_name]
                    .options(num_cpus=num_cpus, num_gpus=num_gpus)
                    .remote(**op_args)
                    for _ in range(op_proc_num)
                ]
            else:
                op_proc_num = calculate_np(op_name, mem_required, num_cpus, None, False)
                logger.info(
                    f"Op {op_name} will be executed on cpu env with op_proc_num {op_proc_num} and use {num_cpus} cpus."
                )
                # 并发actor
                return [
                    OPERATORS.modules[op_name]
                    .options(num_cpus=num_cpus)
                    .remote(**op_args)
                    for _ in range(op_proc_num)
                ]
        else:
            continue


def load_op_names(process_list):
    op_names = []
    for process in process_list:
        op_name, _ = list(process.items())[0]
        op_names.append(op_name)
    return op_names


def replace_if_previous_op_subdirectory(folder_path, op_name):
    folder = Path(folder_path)
    previous_op_name = get_previous_operation(op_name)

    # 检查一级子目录
    for entry in folder.iterdir():
        if entry.is_dir() and entry.name == previous_op_name:
            return os.path.join(folder_path, previous_op_name)

    return folder_path
