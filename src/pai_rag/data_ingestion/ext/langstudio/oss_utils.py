from typing import Optional, Tuple
from urllib import parse


def is_oss_uri(uri: str) -> bool:
    return uri.startswith("oss://")


def parse_oss_uri(oss_uri: str) -> Tuple[str, Optional[str], str]:
    """
    Parse the oss uri to the format of ("<bucket_name>", <endpoint>, <object_key>)
    """
    parsed_result = parse.urlparse(oss_uri)
    if parsed_result.scheme != "oss":
        raise ValueError("require oss uri but get '{}'".format(oss_uri))
    if "." in parsed_result.hostname:
        bucket_name, endpoint = parsed_result.hostname.split(".", 1)
    else:
        bucket_name = parsed_result.hostname
        endpoint = None
    object_key = parsed_result.path
    return bucket_name, endpoint, object_key.lstrip("/")


def standardize_oss_uri(oss_uri: str) -> str:
    """
    Standardize the oss uri to the format of "oss://<bucket_name>/<object_key>"
    """
    bucket_name, _, object_key = parse_oss_uri(oss_uri)
    return f"oss://{bucket_name}/{object_key}"
