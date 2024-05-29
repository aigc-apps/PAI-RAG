import oss2


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
