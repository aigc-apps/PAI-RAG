import os
import ray
import argparse
import time
from loguru import logger
from pai_rag.integrations.readers.pai.pai_data_reader import get_input_files
from pai_rag.tools.data_process.utils.ray_init import init_ray_env
from pai_rag.tools.data_process.actors.parse_actor import ParseActor
from pai_rag.tools.data_process.utils.cuda_util import infer_torch_device


def main(args):
    init_ray_env(args.working_dir)
    input_files = get_input_files(args.data_path)
    _device = args.device or infer_torch_device()
    parser = ParseActor.remote(args.working_dir, args.config_file, _device)
    run_tasks = [parser.load_and_parse.remote(input_file) for input_file in input_files]
    results = ray.get(run_tasks)
    logger.info("Master node completed processing files.")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Write to {args.output_dir}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_file = os.path.join(args.output_dir, f"{timestamp}.jsonl")
    parser.write_to_file.remote(results, save_file)
    logger.info(f"Write to {save_file} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logger.info(f"Init: args: {args}")

    main(args)
