import argparse
from loguru import logger
import ray
import time
from ray.data.datasource.filename_provider import _DefaultFilenameProvider
from pai_rag.tools.data_process.actors.embed_actor import EmbedActor
from pai_rag.tools.data_process.utils.ray_init import init_ray_env, get_num_cpus


def main(args):
    init_ray_env(args.working_dir, args.num_cpus)
    num_cpus_total = get_num_cpus()
    ds = ray.data.read_json(args.data_path)
    logger.info("Embedding nodes started.")
    ds = ds.map_batches(
        EmbedActor,
        fn_constructor_kwargs={
            "working_dir": args.working_dir,
            "config_file": args.config_file,
        },
        num_cpus=args.num_cpus,
        concurrency=int(num_cpus_total / args.num_cpus) - 1,
        batch_size=args.batch_size,
    )
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
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
