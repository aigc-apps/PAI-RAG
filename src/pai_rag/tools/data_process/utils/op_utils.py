from loguru import logger
from pai_rag.tools.data_process.utils.mm_utils import size_to_bytes
from pai_rag.tools.data_process.utils.cuda_utils import get_num_gpus, calculate_np
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from pai_rag.tools.data_process.ops.parser_op import Parser
from pai_rag.tools.data_process.ops.splitter_op import Splitter
from pai_rag.tools.data_process.ops.embed_op import Embedder

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
                    f"Op {op_name} will be executed on cuda env with op_proc: {op_proc} and use {num_cpus} cpus and {num_gpus} GPUs."
                )
                if op_name == "pai_rag_parser":
                    return (
                        Parser.options(
                            num_cpus=num_cpus,
                            num_gpus=num_gpus,
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg
                            ),
                        ).remote(**op_args),
                        pg,
                    )
                elif op_name == "pai_rag_splitter":
                    return (
                        Splitter.options(
                            num_cpus=num_cpus,
                            num_gpus=num_gpus,
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg
                            ),
                        ).remote(**op_args),
                        pg,
                    )
                elif op_name == "pai_rag_embedder":
                    return (
                        Embedder.options(
                            num_cpus=num_cpus,
                            num_gpus=num_gpus,
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg
                            ),
                        ).remote(**op_args),
                        pg,
                    )
            else:
                op_proc = calculate_np(op_name, mem_required, num_cpus, None, False)
                logger.info(
                    f"Op {op_name} will be executed on cpu env with op_proc: {op_proc} and use {num_cpus} cpus."
                )
                pg = placement_group(
                    [{"CPU": num_cpus}] * int(op_proc), strategy="SPREAD"
                )
                ray.get(pg.ready())
                if op_name == "pai_rag_parser":
                    return (
                        Parser.options(
                            num_cpus=num_cpus,
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg
                            ),
                        ).remote(**op_args),
                        pg,
                    )
                elif op_name == "pai_rag_splitter":
                    return (
                        Splitter.options(
                            num_cpus=num_cpus,
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg
                            ),
                        ).remote(**op_args),
                        pg,
                    )
                elif op_name == "pai_rag_embedder":
                    return (
                        Embedder.options(
                            num_cpus=num_cpus,
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg
                            ),
                        ).remote(**op_args),
                        pg,
                    )
        else:
            continue


def load_op_names(process_list):
    op_names = []
    for process in process_list:
        op_name, _ = list(process.items())[0]
        op_names.append(op_name)
    return op_names
