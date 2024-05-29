from pai_rag.utils.oss_cache import OssCache
import click
import logging
from pai_rag.utils.constants import DEFAULT_EASYOCR_MODEL_LOCAL_DIR

logging.getLogger().setLevel(logging.INFO)
DEFAULT_BUCKET = "pai-rag"
DEFAULT_ENDPOINT = "oss-cn-hangzhou.aliyuncs.com"
DEFAULT_EASYOCR_MODEL_PATH = "model/easyocr"


@click.command()
@click.option("-id", "--access_key_id", required=True, help="oss access id")
@click.option("-s", "--access_key_secret", required=True, help="oss access secret")
@click.option("-b", "--bucket", help="oss easyocr model bucket", default=DEFAULT_BUCKET)
@click.option("-e", "--endpoint", help="oss endpoint", default=DEFAULT_ENDPOINT)
@click.option(
    "-mp",
    "--oss_easyocr_models_path",
    help="oss easyocr models path",
    default=DEFAULT_EASYOCR_MODEL_PATH,
)
def download_easyocr_models(
    access_key_id, access_key_secret, bucket, endpoint, oss_easyocr_models_path
):
    oss_config = dict()
    oss_config["access_key_id"] = access_key_id
    oss_config["access_key_secret"] = access_key_secret
    oss_config["bucket_name"] = bucket
    oss_config["endpoint"] = endpoint
    easyocr_model_oss = OssCache(oss_config)
    logging.info("oss connected")
    easyocr_model_oss.download_files(
        oss_easyocr_models_path, DEFAULT_EASYOCR_MODEL_LOCAL_DIR
    )
    logging.info("finished downloading easyocr models")
