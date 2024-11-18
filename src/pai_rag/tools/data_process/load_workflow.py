import ray
import argparse
import json
from loguru import logger
from pai_rag.integrations.readers.pai.pai_data_reader import get_input_files
from pai_rag.tools.data_process.utils.ray_init import init_ray_env


@ray.remote
def write_to_file(results, filename):
    with open(filename, "a", encoding="utf-8") as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + "\n")


@ray.remote
def process(config_file, input_file):
    from pai_rag.tools.data_process.tasks.load_and_parse_doc import (
        load_and_parse_doc_task,
    )

    res = load_and_parse_doc_task(config_file, input_file)
    return res


def main(args):
    _ = init_ray_env(args.working_dir)
    input_files = get_input_files(args.data_path)
    run_tasks = [
        process.remote(args.config_file, input_file) for input_file in input_files
    ]
    results = ray.get(run_tasks)
    logger.info("Master node completed processing files.")
    write_to_file.remote(results, args.output_path)
    logger.info(f"Results written to {args.output_path} asynchronously.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--working_dir", type=str, default=None)
    args = parser.parse_args()

    logger.info(f"Init: args: {args}")

    main(args)
