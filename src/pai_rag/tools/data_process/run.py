import argparse
import yaml
from loguru import logger
from typing import List
from pai_rag.tools.data_process.ops.base_op import OPERATORS
from pai_rag.tools.data_process.ray_executor import RayExecutor
from pai_rag.tools.data_process.utils.compute_resource_utils import (
    enforce_min_requirements,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def extract_parameters(yaml_dict, cfg):
    print("yaml_dict", yaml_dict)
    extracted_params = {key: value for key, value in yaml_dict.items() if key != "op"}
    extracted_params["working_dir"] = cfg.working_dir
    extracted_params["dataset_path"] = cfg.dataset_path
    extracted_params["export_path"] = cfg.export_path
    return extracted_params


def update_op_process(args):
    op_keys = list(OPERATORS.modules.keys())
    logger.info(f"Loading all operation keys: {op_keys}")

    if args.process is None:
        args.process = []

    with open(args.config_file) as file:
        process_cfg = yaml.safe_load(file)
    for i, process_op in enumerate(process_cfg["process"]):
        if process_op["op"] in op_keys:
            args.process.append(process_op["op"])
            args.process[i] = {process_op["op"]: extract_parameters(process_op, args)}

    return args


def process_parser(args):
    op_name = "rag_parser"
    args_dict = args.__dict__
    parser_required_args = {
        key: args_dict[key]
        for key in [
            "dataset_path",
            "export_path",
            "working_dir",
            "cpu_required",
            "mem_required",
            "accelerator",
            "enable_mandatory_ocr",
            "concat_csv_rows",
            "format_sheet_data_to_json",
            "sheet_column_filters",
            "oss_bucket",
            "oss_endpoint",
        ]
    }
    parser_required_args = enforce_min_requirements(op_name, parser_required_args)
    args.process.append(op_name)
    args.process[0] = {op_name: parser_required_args}
    return args


def process_splitter(args):
    op_name = "rag_splitter"
    args_dict = args.__dict__
    splitter_required_args = {
        key: args_dict[key]
        for key in [
            "dataset_path",
            "export_path",
            "working_dir",
            "cpu_required",
            "mem_required",
            "type",
            "chunk_size",
            "chunk_overlap",
            "enable_multimodal",
        ]
    }
    splitter_required_args = enforce_min_requirements(op_name, splitter_required_args)
    args.process.append(op_name)
    args.process[0] = {op_name: splitter_required_args}
    return args


def process_embedder(args):
    op_name = "rag_embedder"
    args_dict = args.__dict__
    embedder_required_args = {
        key: args_dict[key]
        for key in [
            "dataset_path",
            "export_path",
            "working_dir",
            "cpu_required",
            "mem_required",
            "accelerator",
            "source",
            "model",
            "enable_sparse",
            "enable_multimodal",
            "multimodal_source",
            "connection_name",
            "workspace_id",
        ]
    }
    embedder_required_args = enforce_min_requirements(op_name, embedder_required_args)
    args.process.append(op_name)
    args.process[0] = {op_name: embedder_required_args}
    return args


def init_configs():
    """
    initialize the argparse parser and parse configs from one of:
        1. POSIX-style commands line args;
        2. environment variables
        3. hard-coded defaults

    :return: a global cfg object used by the Executor or Analyzer
    """
    parser = argparse.ArgumentParser()
    # shared parameters
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to datasets with optional weights(0.0-1.0), 1.0 as "
        "default. Accepted format:<w1> dataset1-path <w2> dataset2-path "
        "<w3> dataset3-path ...",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="./outputs/hello_world.jsonl",
        help="Path to export and save the output processed dataset. The "
        "directory to store the processed dataset will be the work "
        "directory of this process.",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default="/app",
        help="Path to working dir for ray cluster.",
    )
    parser.add_argument(
        "--cpu_required",
        type=int,
        default=2,
        help="Cpu required for each rag operator.",
    )
    parser.add_argument(
        "--mem_required",
        type=int,
        default=2,
        help="Memory(GB) required for each rag operator.",
    )
    parser.add_argument(
        "--process", default=[], help="list of operator processes to run"
    )
    # Only used for multi-operators mode
    parser.add_argument(
        "--config_file",
        help="Path to a dj basic configuration file.",
        type=str,
        default=None,
    )
    # Only used for single-operator mode
    parser.add_argument("--operator", default=None, help="Choose a rag operator to run")

    # arguments for rag_parser
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Accelerator type for rag_parser and rag_embedder operator.",
    )
    parser.add_argument(
        "--enable_mandatory_ocr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to enable mandatory OCR for rag_parser operator.",
    )
    parser.add_argument(
        "--concat_csv_rows",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to concat csv rows for rag_parser operator.",
    )
    parser.add_argument(
        "--format_sheet_data_to_json",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to format sheet data to json for rag_parser operator.",
    )
    parser.add_argument(
        "--sheet_column_filters",
        type=List[str],
        default=[],
        help="Column filters for rag_parser operator.",
    )
    parser.add_argument(
        "--oss_bucket",
        type=str,
        default=None,
        help="OSS Bucket for rag_parser operator.",
    )
    parser.add_argument(
        "--oss_endpoint",
        type=str,
        default="oss-cn-hangzhou.aliyuncs.com",
        help="OSS Endpoint for rag_parser operator.",
    )

    # arguments for rag_splitter
    parser.add_argument(
        "--type",
        type=str,
        default="Token",
        help="Split type for rag_splitter operator.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Chunk size for rag_splitter operator.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=20,
        help="Chunk overlap for rag_splitter operator.",
    )
    parser.add_argument(
        "--enable_multimodal",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to enable multimodal for rag_splitter and rag_embedder operator.",
    )

    # arguments for rag_embedder
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        help="Embedding model source for rag_embedder operator.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge-m3",
        help="Embedding model name for rag_embedder operator.",
    )
    parser.add_argument(
        "--enable_sparse",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to enable sparse for rag_embedder operator.",
    )
    parser.add_argument(
        "--multimodal_source",
        type=str,
        default="cnclip",
        help="Multi-modal embedding model source for rag_embedder operator.",
    )
    parser.add_argument(
        "--connection_name",
        type=str,
        default=None,
        help="Langstudio connection for rag_embedder operator.",
    )
    parser.add_argument(
        "--workspace_id",
        type=str,
        default=None,
        help="PAI workspace id for rag_embedder operator.",
    )

    args = parser.parse_args()

    # Determine which way to run with rag operators [config_file, cmd_args]
    if args.config_file is not None:
        args = update_op_process(args)
    elif args.operator == "rag_parser":
        args = process_parser(args)
    elif args.operator == "rag_splitter":
        args = process_splitter(args)
    elif args.operator == "rag_embedder":
        args = process_embedder(args)
    else:
        parser.print_help()

    logger.info(f"Final args: {args}")
    return args


@logger.catch(reraise=True)
def main():
    cfg = init_configs()
    executor = RayExecutor(cfg)
    executor.run()


if __name__ == "__main__":
    main()
