import oss2
import os


def calculate_file_md5(file_path: str, prefix: str) -> str:
    """
    calculate oss MD5校验
    """
    with open(file_path, "rb") as file:
        content = file.read()

    """根据实际内容计算上传内容对应的MD5。"""
    content_md5 = oss2.utils.content_md5(content)

    """上传请求中携带'Content-MD5'的header，服务器会校验上传内容的MD5，用于保证上传内容的完整性和正确性。"""
    headers = dict()
    headers["Content-MD5"] = content_md5

    """上传文件"""
    key = prefix + content_md5
    return key


def check_and_set_oss_auth(config_snapshot):
    """
    check snapshot config and set oss auth
    """
    if config_snapshot["oss_store"].get("ak") and config_snapshot["oss_store"].get(
        "sk"
    ):
        if "***" not in config_snapshot["oss_store"].get("ak"):
            os.environ["OSS_ACCESS_KEY_ID"] = config_snapshot["oss_store"].get("ak")
        if "***" not in config_snapshot["oss_store"].get("sk"):
            os.environ["OSS_ACCESS_KEY_SECRET"] = config_snapshot["oss_store"].get("sk")
        del config_snapshot["oss_store"]["ak"]
        del config_snapshot["oss_store"]["sk"]
    return config_snapshot


def get_oss_auth(config_dict_value):
    """
    get oss auth and return to config dict
    """
    oss_auth_ak = os.getenv("OSS_ACCESS_KEY_ID")
    oss_auth_sk = os.getenv("OSS_ACCESS_KEY_SECRET")
    if oss_auth_ak:
        config_dict_value["RAG"]["oss_store"]["ak"] = oss_auth_ak
    else:
        config_dict_value["RAG"]["oss_store"]["ak"] = None
    if oss_auth_sk:
        config_dict_value["RAG"]["oss_store"]["sk"] = oss_auth_sk
    else:
        config_dict_value["RAG"]["oss_store"]["sk"] = None
    return config_dict_value
