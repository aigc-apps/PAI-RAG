from loguru import logger
from pai_rag.tools.data_process.ray_executor import RayExecutor
import argparse
from pai_rag.tools.data_process.ops.base_op import OPERATORS
import yaml


def extract_parameters(yaml_dict, cfg):
    extracted_params = {key: value for key, value in yaml_dict.items() if key != "op"}
    extracted_params["working_dir"] = cfg.working_dir
    extracted_params["dataset_path"] = cfg.dataset_path
    extracted_params["export_path"] = cfg.export_path
    return extracted_params


def update_op_process(cfg):
    op_keys = list(OPERATORS.modules.keys())

    if cfg.process is None:
        cfg.process = []

    with open(cfg.config) as file:
        process_cfg = yaml.safe_load(file)
    for i, process_op in enumerate(process_cfg["process"]):
        if process_op["op"] in op_keys:
            cfg.process.append(process_op["op"])
            cfg.process[i] = {process_op["op"]: extract_parameters(process_op, cfg)}

    return cfg


def init_configs():
    """
    initialize the jsonargparse parser and parse configs from one of:
        1. POSIX-style commands line args;
        2. config files in yaml (json and jsonnet supersets);
        3. environment variables
        4. hard-coded defaults

    :return: a global cfg object used by the Executor or Analyzer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to a dj basic configuration file.",
        required=True,
    )
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
        default="/home/xiaowen/xiaowen/github_code/PAI-RAG",
        help="Path to working dir for ray cluster.",
    )
    parser.add_argument("--process", type=int, default=None, help="name of processes")
    cfg = parser.parse_args()
    cfg = update_op_process(cfg)
    return cfg


@logger.catch(reraise=True)
def main():
    cfg = init_configs()
    executor = RayExecutor(cfg)
    executor.run()


if __name__ == "__main__":
    main()
