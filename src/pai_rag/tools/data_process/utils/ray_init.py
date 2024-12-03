import ray
from loguru import logger
import os

os.environ["RAY_DEDUP_LOGS"] = "0"


def init_ray_env(working_dir, num_cpus: int = 1):
    ray.init(runtime_env={"working_dir": working_dir})
    logger.info(
        """This cluster consists of
        {} nodes in total
        {} CPU resources in total
        {} CPU per actor
        {} concurrency
    """.format(
            len(ray.nodes()),
            ray.cluster_resources()["CPU"],
            num_cpus,
            int(ray.cluster_resources()["CPU"] / num_cpus),
        )
    )


def get_num_workers():
    return len(ray.nodes())


def get_num_cpus():
    return ray.cluster_resources()["CPU"]
