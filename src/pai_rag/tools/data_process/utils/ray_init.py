import ray
from loguru import logger
import os

os.environ["RAY_DEDUP_LOGS"] = "0"


def init_ray_env(working_dir, num_cpus_per_actor: int = 1):
    ray.init(runtime_env={"working_dir": working_dir})
    logger.info(
        """This cluster consists of
        {} nodes in total
        {} CPU resources in total
        {} CPU needed per actor
        {} concurrency
    """.format(
            len(ray.nodes()),
            ray.cluster_resources()["CPU"],
            num_cpus_per_actor,
            get_concurrency(num_cpus_per_actor),
        )
    )


def get_concurrency(num_cpus_per_actor: int = 1):
    num_cpus_total = ray.cluster_resources()["CPU"] or 1
    if num_cpus_total > 1:
        return int((num_cpus_total - 2) / num_cpus_per_actor)
    else:
        return 1
