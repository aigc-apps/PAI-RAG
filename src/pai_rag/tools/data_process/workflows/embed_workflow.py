import argparse
from loguru import logger
import ray
import time
from ray.data.datasource.filename_provider import _DefaultFilenameProvider
from pai_rag.tools.data_process.actors.embed_actor import EmbedActor
from pai_rag.tools.data_process.utils.ray_init import init_ray_env, get_concurrency
from pai_rag.tools.data_process.utils.cuda_util import infer_torch_device


def main(args):
    init_ray_env(args.working_dir, args.num_cpus)
    _device = args.device or infer_torch_device()
    ds = ray.data.read_json(args.data_path)
    logger.info("Embedding nodes started.")
    ds = ds.map_batches(
        EmbedActor,
        fn_constructor_kwargs={
            "working_dir": args.working_dir,
            "config_file": args.config_file,
        },
        concurrency=get_concurrency(args.num_cpus),
        batch_size=args.batch_size,
    )
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
    logger.info(f"Write to {args.output_dir} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
