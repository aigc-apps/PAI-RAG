from loguru import logger
from pai_rag.tools.data_process.ray_executor import RayExecutor
from jsonargparse import ActionConfigFile, ArgumentParser
from typing import List, Optional


def init_configs(args: Optional[List[str]] = None):
    """
    initialize the jsonargparse parser and parse configs from one of:
        1. POSIX-style commands line args;
        2. config files in yaml (json and jsonnet supersets);
        3. environment variables
        4. hard-coded defaults

    :param args: list of params, e.g., ['--conifg', 'cfg.yaml'], default None.
    :return: a global cfg object used by the Executor or Analyzer
    """
    parser = ArgumentParser(default_env=True, default_config_files=None)

    parser.add_argument(
        "--config",
        action=ActionConfigFile,
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
        "--working_dir", type=str, default="/home/xiaowen/xiaowen/github_code/PAI-RAG"
    )


@logger.catch(reraise=True)
def main():
    cfg = init_configs()
    executor = RayExecutor(cfg)
    executor.run()


if __name__ == "__main__":
    main()
