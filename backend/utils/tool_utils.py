from typing import Any, Dict
def to_openai_tool(tool_metadata) -> Dict[str, Any]:
        """To OpenAI tool."""
        return {
            "type": "function",
            "function": {
                "name": tool_metadata.name,
                "description": tool_metadata.description,
                "parameters": tool_metadata.get_parameters_dict(),
            },
        }





# async def get_binary_content_from_oss_url(oss_url):
#     """
#     异步从OSS URL获取二进制内容

#     Args:
#         oss_url: OSS文件的URL

#     Returns:
#         bytes: 文件的二进制内容

#     Raises:
#         IOError: 当下载失败时抛出异常
#         aiohttp.ClientError: 当网络请求失败时抛出异常
#     """
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.get(oss_url) as response:
#                 if response.status != 200:
#                     raise IOError(f"Failed to download file from {oss_url}. Status code: {response.status}")

#                 return await response.read()

#     except aiohttp.ClientError as e:
#         logger.error(f"Network error when downloading from {oss_url}: {e}")
#         raise IOError(f"Network error when downloading from {oss_url}: {e}")
#     except asyncio.TimeoutError as e:
#         logger.error(f"Timeout when downloading from {oss_url}: {e}")
#         raise IOError(f"Timeout when downloading from {oss_url}: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error when downloading from {oss_url}: {e}")
#         raise IOError(f"Unexpected error when downloading from {oss_url}: {e}")
