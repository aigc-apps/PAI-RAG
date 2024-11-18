import argparse
import ray
from loguru import logger
from ray.data.datasource.filename_provider import _DefaultFilenameProvider
from pai_rag.tools.data_process.tasks.split_node import split_node_task
from pai_rag.tools.data_process.utils.ray_init import init_ray_env


def main(args):
    NUM_WORKERS = init_ray_env(args.working_dir)
    ds = ray.data.read_json(args.data_path)
    logger.info("Splitting nodes started.")
    ds = ds.flat_map(
        split_node_task,
        fn_kwargs={"config_file": args.config_file},
        concurrency=NUM_WORKERS,
    )
    logger.info("Splitting nodes completed.")
    ds = ds.repartition(1)
    logger.info(f"Write to {args.output_path}")
    ds.write_json(
        args.output_path,
        filename_provider=_DefaultFilenameProvider(file_format="jsonl"),
        force_ascii=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--working_dir", type=str, default=None)
    args = parser.parse_args()

    logger.info(f"Init: args: {args}")

    main(args)
