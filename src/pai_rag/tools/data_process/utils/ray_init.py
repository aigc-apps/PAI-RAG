import ray
from loguru import logger
import os

os.environ["RAY_DEDUP_LOGS"] = "0"


def init_ray_env(working_dir):
    ray.init(runtime_env={"working_dir": working_dir})
    logger.info(
        """This cluster consists of
        {} nodes in total
        {} CPU resources in total
    """.format(
            len(ray.nodes()), ray.cluster_resources()["CPU"]
        )
    )
    NUM_WORKERS = len(ray.nodes())
    return NUM_WORKERS
