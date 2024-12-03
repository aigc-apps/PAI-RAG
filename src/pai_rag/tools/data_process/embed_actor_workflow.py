import argparse
from loguru import logger
import ray
import time
from ray.data.datasource.filename_provider import _DefaultFilenameProvider
from pai_rag.tools.data_process.actors.embed_actor import EmbedActor
from pai_rag.tools.data_process.utils.ray_init import init_ray_env, get_num_workers


def main(args):
    init_ray_env(args.working_dir)
    num_workers = get_num_workers()
    ds = ray.data.read_json(args.data_path)
    logger.info("Embedding nodes started.")
    ds = ds.map(EmbedActor(config_file=args.config_file), concurrency=num_workers)
    logger.info("Embedding nodes completed.")
    logger.info(f"Write to {args.output_dir}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ds = ds.repartition(1)
    ds.write_json(
        args.output_dir,
        filename_provider=_DefaultFilenameProvider(
            dataset_uuid=timestamp, file_format="jsonl"
        ),
        force_ascii=False,
    )
    logger.info("Write completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
