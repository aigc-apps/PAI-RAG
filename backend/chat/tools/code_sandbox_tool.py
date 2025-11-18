# tools/code_sandbox_tool.py
import json
import asyncio
import aiohttp
import json5
from typing import Optional, Union
import re
import os
from functools import wraps
from pairag.file.utils.image_utils import compress_image_if_needed
from pairag.file.store.file_store_helper import file_store
from llama_index.core.tools import FunctionTool
import uuid
from io import BytesIO
from aiohttp import FormData
from loguru import logger
from utils.http_session import HttpSessionShared
from chat.tools.code_sandbox_exceptions import (
    CodeSandboxEmptyCodeException,
    CodeSandboxNotInitializedException,
    CodeSandboxTimeoutException,
    CodeSandboxHTTPException,
    CodeSandboxAPIException,
    CodeSandboxExecutionException,
)

DEFAULT_CODE_SANDBOX_DIR_PATH = '/home/user'
DEFAULT_CODE_SANDBOX_SYSTEM_FILES = ('.bash_logout', '.bashrc', '.profile')
TRIPLE_QUOTE_PATTERN = re.compile(r'```[^\n]*\n(.+?)```', re.DOTALL)
XML_CODE_PATTERN = re.compile(r'<code>(.*?)</code>', re.DOTALL)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
ext_pattern = '|'.join(ext[1:] for ext in IMAGE_EXTENSIONS)
MARKDOWN_IMG_PATTERN = re.compile(rf'(!\[[^\]]*\]\()([^\)]+\.(?:{ext_pattern}))(\))', re.IGNORECASE)


async def get_http_client_session(timeout: int = 600):
    """异步生成器函数，用于获取 HTTP client session"""
    session = await HttpSessionShared.ensure_session()
    yield session


def with_http_client_session(timeout: int = 600):
    """装饰器，自动管理 HTTP client session 的生命周期"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            session = await HttpSessionShared.ensure_session()
            kwargs["session"] = session
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class CodeSandboxTool:
    def __init__(
        self,
        aliyun_id: str,
        interpreter_id: str,
        timeout_default: int = 50,
        enabled: bool = False,
    ):
        self.enabled = enabled
        self.aliyun_id = aliyun_id
        self.interpreter_id = interpreter_id
        self.timeout_default = timeout_default
        self.base_url = f"https://{self.aliyun_id}.agentrun-data.cn-hangzhou.aliyuncs.com/2025-09-10/agents/code-interpreters/{self.interpreter_id}"

    async def _get_session(self, timeout: int = 600) -> aiohttp.ClientSession:
        """获取 HTTP session，使用共享的 session"""
        return await HttpSessionShared.ensure_session()

    async def acreate_session_and_context(self):
        # uuid v4,不能是数字开头
        s = str(uuid.uuid4())
        session_id = "a" + s[1:]
        headers_with_session = {
            "Content-Type": "application/json",
            "X-AgentRun-Session-ID": session_id,
        }

        payload = {
            "name": "data-analysis",
            "language": "python",
            "config": {}
        }

        url = f"{self.base_url}/contexts"
        session = await self._get_session()
        try:
            async with session.post(
                url,
                headers=headers_with_session,
                data=json.dumps(payload)
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Failed to create context: {response.status} {text}")
                    response.raise_for_status()
                result = await response.json()
        except Exception:
            logger.exception("Error in acreate_session_and_context")
            raise
        if "data" in result and "id" in result["data"]:
            context_id = result["data"]["id"]
        else:
            raise Exception("Failed to create context")
        return session_id, context_id


    async def aupload_data_file_to_sandbox(self, file_content: bytes, file_name: str, session_id: str):
        if not session_id:
            logger.error("Session ID not set. Cannot upload file to sandbox.")
            return None

        headers_with_session = {
            "X-AgentRun-Session-ID": session_id,
        }
        form = FormData()
        form.add_field(
        'file',
        file_content,
        filename=file_name,
        content_type='application/octet-stream'
    )

        path = os.path.join(DEFAULT_CODE_SANDBOX_DIR_PATH, file_name)
        params = {'path': path}

        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/files",
                headers=headers_with_session,
                data=form,
                params=params
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Upload failed: {response.status} {text}")
                    response.raise_for_status()
                return await response.json()
        except Exception:
            logger.exception(f"Failed to upload file {file_name} to sandbox")
            return None

    async def adownload_result_from_sandbox(self, file_path: str, session_id: str):
        if not session_id:
            logger.error("Session ID not set. Cannot download file.")
            return None

        headers_with_session = {
            "X-AgentRun-Session-ID": session_id,
        }

        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/files",
                headers=headers_with_session,
                params={'path': file_path}
            ) as response:
                if not response.ok:
                    logger.error(f"Download failed: {response.status}")
                    response.raise_for_status()
                return await response.read()  # 返回 bytes
        except Exception:
            logger.exception(f"Failed to download file {file_path}")
            return None

    async def alist_code_sandbox_dir_file_paths(self, file_dir_path: str, session_id: str):
        if not session_id:
            logger.error("Session ID not set. Cannot list dir files.")
            return json.dumps({"paths": ""}, ensure_ascii=False)

        headers_with_session = {
            "X-AgentRun-Session-ID": session_id,
        }

        result = None
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/filesystem",
                headers=headers_with_session,
                params={'path': file_dir_path}
            ) as response:
                if not response.ok:
                    logger.error(f"List files failed: {response.status}")
                    response.raise_for_status()
                result = await response.json()
        except Exception:
            logger.exception("Error listing sandbox files")
            return json.dumps({"paths": ""}, ensure_ascii=False)

        sandbox_file_paths = []
        if result and 'data' in result and 'entries' in result["data"]:
            sandbox_file_paths = [
                item['path'] for item in result["data"]['entries']
                if item['name'] not in DEFAULT_CODE_SANDBOX_SYSTEM_FILES
            ]
        return json.dumps({"data": ','.join(sandbox_file_paths)}, ensure_ascii=False)

    def _extract_code(self, params: Union[str, dict]) -> str:
        """
        从输入参数中提取代码字符串。

        支持多种输入格式：
        1. 字符串格式：可以是 JSON5 格式的字符串，包含 'code' 或 'raw' 字段
        2. 字典格式：包含 'code' 或 'raw' 字段的字典

        支持多种代码包裹格式：
        1. 三重引号格式：```python\ncode``` 或 ```\ncode```
        2. XML 标签格式：<code>code</code>
        3. 原始字符串格式：直接是代码字符串

        Args:
            params: 字符串或字典，包含要执行的代码

        Returns:
            提取并清理后的代码字符串。如果解析失败，返回原始字符串（去除首尾空格）或空字符串。
        """
        try:
            # 如果是字符串，先尝试解析为 JSON5 格式
            if isinstance(params, str):
                params = json5.loads(params)
            # 从字典中获取代码，优先使用 'code' 字段，其次使用 'raw' 字段
            code = params.get('code', '') or params.get('raw', '')
            # 尝试匹配三重引号格式（```...```）
            triple_match = TRIPLE_QUOTE_PATTERN.search(code)
            if triple_match:
                code = triple_match.group(1)
            else:
                # 尝试匹配 XML 标签格式（<code>...</code>）
                xml_match = XML_CODE_PATTERN.search(code)
                if xml_match:
                    code = xml_match.group(1)
            return code.strip()
        except Exception:
            # 如果解析失败，fallback 到原始字符串或返回空字符串
            if isinstance(params, str):
                return params.strip()
            return ""

    async def areplace_code_sandbox_image_paths(self, final_result: str, session_id: str) -> str:

        replacements = []

        for match in MARKDOWN_IMG_PATTERN.finditer(final_result):
            prefix = match.group(1)
            local_filepath = match.group(2)
            suffix = match.group(3)

            file_name = os.path.basename(local_filepath)
            _, ext = os.path.splitext(file_name.lower())
            if ext not in IMAGE_EXTENSIONS:
                continue

            try:
                local_filepath_full = os.path.join(DEFAULT_CODE_SANDBOX_DIR_PATH, file_name)

                image_blob = await self.adownload_result_from_sandbox(local_filepath_full, session_id)
                if not isinstance(image_blob, bytes):
                    logger.warning(f"Download did not return bytes for {file_name}")
                    continue

                image_file = BytesIO(image_blob)

                if ext != '.svg':
                    image_file = compress_image_if_needed(image_file)
                    if not image_file:
                        logger.warning(f"Image compression failed for {file_name}, skipping.")
                        continue

                destination_file_path = f"default_chat_docs/docs/{file_name}"
                file_store.save(
                    file=image_file,
                    file_path=destination_file_path,
                )
                url = file_store.get_url(destination_file_path)
                replacement = f"{prefix}{url}{suffix}"
                replacements.append((match.start(), match.end(), replacement))

            except Exception as e:
                logger.error(f"Failed to process image {file_name}: {e}")
                continue

        result = final_result
        for start, end, repl in reversed(replacements):
            result = result[:start] + repl + result[end:]

        return result

    async def aexecute(self, code: str, timeout: Optional[int] = None, session_id: str = None, context_id: str = None) -> str:
        code = self._extract_code(code)
        if not code:
            logger.error("Empty or invalid code provided")
            raise CodeSandboxEmptyCodeException("Empty or invalid code provided")
        if not session_id or not context_id:
            logger.error("Session or context not initialized")
            raise CodeSandboxNotInitializedException("Session or context not initialized")

        actual_timeout = timeout or self.timeout_default
        headers = {
            "Content-Type": "application/json",
            "X-AgentRun-Session-ID": session_id,
        }

        payload = {"code": code}
        url = f"{self.base_url}/contexts/{context_id}/execute"

        # aiohttp 的 timeout 是总超时（包括连接+读取）
        timeout_obj = aiohttp.ClientTimeout(total=actual_timeout + 5)
        result = None
        # 使用共享 session，但请求时指定动态 timeout
        session = await HttpSessionShared.ensure_session()
        try:
            async with session.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout_obj
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Execute failed: {response.status} {text}")
                    raise CodeSandboxHTTPException(f"HTTP {response.status}: {text}")
                result = await response.json()
        except asyncio.TimeoutError:
            logger.error("Execution timed out")
            raise CodeSandboxTimeoutException("Execution timed out")
        except (CodeSandboxHTTPException, CodeSandboxTimeoutException):
            raise
        except Exception as e:
            logger.exception("Error during code execution")
            raise CodeSandboxExecutionException(f"Error during code execution: {str(e)}") from e

        if result is None or result.get("code") != "SUCCESS":
            error_msg = f"API returned non-success code: {result.get('code') if result else 'None'}"
            logger.error(error_msg)
            raise CodeSandboxAPIException(error_msg)

        results = result.get("data", {}).get("results", [])
        stdout_lines = []
        stderr_lines = []
        error_lines = []
        timed_out = False

        for item in results:
            typ = item.get("type")
            if typ == "stdout":
                stdout_lines.append(item.get("text", ""))
            elif typ == "stderr":
                stderr_lines.append(item.get("text", ""))
            elif typ == "error":
                error_lines.append(f"{item.get('name', '')}: {item.get('value', '')}")
            elif typ == "timeout":
                timed_out = True

        parts = []
        if stdout_lines:
            parts.append("stdout:\n" + "".join(stdout_lines).rstrip())
        if stderr_lines:
            parts.append("stderr:\n" + "".join(stderr_lines).rstrip())
        if error_lines:
            parts.append("error:\n" + "\n".join(error_lines).rstrip())
        if timed_out or any("TimeoutError" in line for line in stderr_lines):
            logger.error("Execution timed out in results")
            raise CodeSandboxTimeoutException("Execution timed out")

        final_result = "\n".join(parts).strip()
        final_result = await self.areplace_code_sandbox_image_paths(final_result, session_id)
        return final_result if final_result else "Finished execution, but no result."

    async def aget_list_directory_files_tool(self, session_id: str):
        async def _wrapped_list_files(file_dir_path: str, session_id: str):
            return await self.alist_code_sandbox_dir_file_paths(file_dir_path, session_id)

        description = f"""
        列出 CodeSandbox 中指定目录（如 {DEFAULT_CODE_SANDBOX_DIR_PATH}）下的所有文件路径。
        会自动过滤掉 {DEFAULT_CODE_SANDBOX_SYSTEM_FILES} 等系统文件。
        输入应为一个目录路径字符串。
        """

        return FunctionTool.from_defaults(
            async_fn=_wrapped_list_files,
            name="list-sandbox-files",
            description=description,
        )
