import re


def get_region_from_endpoint(endpoint: str) -> str:
    """Extract region from OSS endpoint.

    Examples:
        oss-cn-hangzhou.aliyuncs.com -> cn-hangzhou
        oss-cn-hangzhou-internal.aliyuncs.com -> cn-hangzhou
        oss-us-east-1.aliyuncs.com -> us-east-1
    """
    match = re.search(r"oss-([a-z0-9-]+?)(?:-internal)?\.aliyuncs\.com", endpoint)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract region from endpoint: {endpoint}")
