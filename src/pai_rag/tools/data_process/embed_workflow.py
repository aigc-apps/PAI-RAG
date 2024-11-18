import argparse
from loguru import logger
import ray
from ray.data.datasource.filename_provider import _DefaultFilenameProvider
from pai_rag.tools.data_process.tasks.embed_node import embed_node_task
from pai_rag.tools.data_process.utils.ray_init import init_ray_env


def main(args):
    NUM_WORKERS = init_ray_env(args.working_dir)
    ds = ray.data.read_json(args.data_path)
    logger.info("Starting to embedding..")
    ds = ds.map(
        embed_node_task,
        fn_kwargs={"config_file": args.config_file},
        concurrency=NUM_WORKERS,
    )
    logger.info(f"Embedding completed with blocks: {ds.num_blocks()}..")
    ds = ds.repartition(1)
    logger.info(f"Starting to write to {args.output_path}")
    ds.write_json(
        args.output_path,
        filename_provider=_DefaultFilenameProvider(file_format="jsonl"),
        force_ascii=False,
    )
    logger.info("Write completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--working_dir", type=str, default=None)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
