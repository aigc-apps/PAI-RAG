# tools/code_sandbox_tool.py
import json
import asyncio
import aiohttp
import json5
from typing import Optional, Union, List
import re
import os
from functools import wraps
from pairag.file.utils.image_utils import compress_image_if_needed
from pairag.file.store.file_store_helper import file_store
from service.file.file_resource_service import FileResourceService
from llama_index.core.tools import FunctionTool
from io import BytesIO
from loguru import logger
from utils.http_session import HttpSessionShared
from tools.code.code_sandbox_exceptions import (
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
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.heic', '.webp'}
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
        interpreter_name: str,
        tenant_id: str,
        timeout_default: int = 50,
        enabled: bool = False,
        code_sandbox_attachments_ids: list = None,
        file_service: FileResourceService = None,
        api_key: str = None,
    ):
        self.enabled = enabled
        self.aliyun_id = aliyun_id
        self.interpreter_id = interpreter_id
        self.interpreter_name = interpreter_name
        self.timeout_default = timeout_default
        self.api_key = api_key
        # 使用 aliyun_id 构建 base_url，与用户提供的示例一致
        self.base_url = f"https://{self.aliyun_id}.agentrun-data.cn-hangzhou.aliyuncs.com"
        self._sandbox_initialized = False
        self._sandbox_id = None
        self._sandbox_context_id = None
        self._code_sandbox_attachments_ids = code_sandbox_attachments_ids or []
        self.file_service = file_service
        self.tenant_id = tenant_id

    def _get_headers(self) -> dict:
        """生成包含 X-API-Key 的请求头"""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _ensure_sandbox_initialized(self):
        """确保 sandbox 已初始化，如果未初始化则进行初始化"""
        if not self._sandbox_initialized:
            try:
                self._sandbox_id, self._sandbox_context_id = await self.initialize_sandbox_with_attachments(self._code_sandbox_attachments_ids)
                self._sandbox_initialized = True
                logger.info("Sandbox initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize sandbox: {e}")
                raise CodeSandboxNotInitializedException("Failed to initialize sandbox")

    async def initialize_sandbox_with_attachments(self, code_sandbox_attachments_ids: list = None):
        """初始化 sandbox 并上传附件"""
        # 1. 创建sandbox实例
        sandbox_id = await self.acreate_sandbox_instance()

        # 1.5. 检查sandbox实例健康状态，如果是新创建的实例，等待其完全启动
        # 必须等待 sandbox 完全就绪（Jupyter 服务启动，状态为 ok）
        await self.acheck_sandbox_health(sandbox_id, max_wait_seconds=60)

        # 2. 创建context
        context_id = await self.acreate_context(sandbox_id)


        # 3. 如果有附件，上传文件
        if code_sandbox_attachments_ids:
            logger.info(f"[Model] uploading {len(code_sandbox_attachments_ids)} code sandbox attachments.")
            await self.aupload_files_to_code_sandbox(file_ids=code_sandbox_attachments_ids, sandbox_id=sandbox_id)
            logger.info("[Model] Code sandbox ready and files uploaded.")
        else:
            logger.info("[Model] Code sandbox ready.")

        return sandbox_id, context_id

    async def _get_session(self, timeout: int = 600) -> aiohttp.ClientSession:
        """获取 HTTP session，使用共享的 session"""
        return await HttpSessionShared.ensure_session()

    async def acreate_sandbox_instance(self):
        """创建 sandbox 实例"""
        headers = self._get_headers()
        payload = {
            "templateName": self.interpreter_name,
        }
        url = f"{self.base_url}/sandboxes"
        session = await self._get_session()

        try:
            async with session.post(
                url,
                headers=headers,
                json=payload
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Failed to create sandbox instance: {response.status} {text}")
                    response.raise_for_status()
                result = await response.json()
        except Exception as e:
            logger.exception("Error in acreate_sandbox_instance")
            raise CodeSandboxAPIException(f"Failed to create sandbox instance: {e}")

        if "data" in result and "sandboxId" in result["data"]:
            sandbox_id = result["data"]["sandboxId"]
            logger.info(f"Sandbox instance created successfully: {sandbox_id}")
            return sandbox_id
        else:
            raise CodeSandboxAPIException("Failed to create sandbox instance: invalid response format")

    async def _fetch_sandbox_health_status(self, sandbox_id: str) -> str:
        """
        获取 sandbox 健康状态（单次请求）

        Args:
            sandbox_id: sandbox ID

        Returns:
            健康检查结果字符串

        Raises:
            CodeSandboxAPIException: 当请求失败或响应格式无效时
        """
        headers = self._get_headers()
        url = f"{self.base_url}/sandboxes/{sandbox_id}/health"
        session = await self._get_session()

        try:
            async with session.get(url, headers=headers) as response:
                text = await response.text()
                result = json.loads(text)
                if isinstance(result, dict) and "status" in result:
                    return result.get("status")
                else:
                    raise ValueError(f"Invalid health check response: missing 'status' field in {result}")
        except Exception as e:
            logger.exception("Error in _fetch_sandbox_health_status")
            raise CodeSandboxAPIException(f"Failed to check sandbox health: {e}")

    async def acheck_sandbox_health(self, sandbox_id: str, max_wait_seconds: int = 60):
        """
        检查 sandbox 健康状态，必须等待状态为 ok 才返回

        Args:
            sandbox_id: sandbox ID
            max_wait_seconds: 最大等待时间（秒）

        Returns:
            健康检查结果字典，status 字段为 "ok"

        Raises:
            CodeSandboxAPIException: 当超时或健康检查失败时
        """
        try:
            async with asyncio.timeout(max_wait_seconds):
                while True:
                    status = await self._fetch_sandbox_health_status(sandbox_id)

                    if status == "ok":
                        logger.info(f"Sandbox health check passed: {status}")
                        return status
                    else:
                        # 状态不是 "ok"，等待后重试
                        logger.info(f"Sandbox not ready (status: {status}), waiting 5s before retry...")
                        await asyncio.sleep(5)
                        continue

        except TimeoutError:
            logger.error(f"Sandbox health check timeout after {max_wait_seconds}s")
            raise CodeSandboxAPIException(f"Sandbox health check timeout after {max_wait_seconds}s")

    async def adelete_sandbox_instance(self, sandbox_id: str=None):
        sandbox_id = sandbox_id or self._sandbox_id
        if not sandbox_id:
            logger.info("sandbox_id not set. Cannot delete sandbox instance.")
            return None
        headers = self._get_headers()
        url = f"{self.base_url}/sandboxes/{sandbox_id}"
        session = await self._get_session()
        try:
            async with session.delete(
                url,
                headers=headers
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Failed to delete sandbox instance: {response.status} {text}")
                    response.raise_for_status()
                else:
                    result = await response.json()
                    self._sandbox_id = None
                    self._sandbox_context_id = None
                    logger.info(f"Sandbox instance deleted successfully: sandboxId={sandbox_id}, status={result.get('status', 'N/A')}")
        except Exception as e:
            logger.exception("Error in adelete_sandbox_instance")
            raise CodeSandboxAPIException(f"Failed to delete sandbox instance: {e}")

    async def acreate_context(self, sandbox_id: str, cwd=None):
        headers = self._get_headers()
        payload = {
            "language": "python",
        }
        if cwd:
            payload["cwd"] = cwd

        url = f"{self.base_url}/sandboxes/{sandbox_id}/contexts"
        session = await self._get_session()
        try:
            async with session.post(
                url,
                headers=headers,
                json=payload
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Failed to create context: {response.status} {text}")
                    response.raise_for_status()
                result = await response.json()
        except Exception as e:
            logger.exception("Error in create context")
            raise CodeSandboxAPIException(f"Failed to create context: {e}")

        # Handle multiple response formats
        context_id = None
        if "data" in result and "id" in result["data"]:
            context_id = result["data"]["id"]
        elif "id" in result:
            context_id = result["id"]

        if context_id:
            logger.info(f"Context created successfully: {context_id}")
            return context_id
        else:
            logger.error(f"Failed to create context: unexpected response format. Response: {result}")
            raise CodeSandboxAPIException(f"Failed to create context: invalid response format. Response: {result}")


    async def aupload_data_file_to_sandbox(self, file_content: bytes, file_name: str, sandbox_id: str = None):
        sandbox_id = sandbox_id or self._sandbox_id
        if not sandbox_id:
            logger.error("sandbox_id not set. Cannot upload file to sandbox.")
            return None

        # 构建 multipart/form-data 请求
        # 使用 FormData 来自动处理 multipart/form-data 格式
        form_data = aiohttp.FormData()
        form_data.add_field('file',
                           file_content,
                           filename=file_name,
                           content_type='application/octet-stream')
        path = os.path.join(DEFAULT_CODE_SANDBOX_DIR_PATH, file_name)
        form_data.add_field('path', path)

        headers = self._get_headers()
        # 对于 multipart/form-data，移除 Content-Type，让 aiohttp 自动设置
        if "Content-Type" in headers:
            headers.pop("Content-Type")
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/sandboxes/{sandbox_id}/filesystem/upload",
                data=form_data,
                headers=headers
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Upload failed: {response.status} {text}")
                    response.raise_for_status()
                result = await response.json()
                logger.info(f"File {file_name} uploaded successfully to {result.get('path', 'N/A')}")
                return result
        except Exception as e:
            logger.exception(f"Failed to upload file {file_name} to sandbox: {e}")
            return None

    async def adownload_result_from_sandbox(self, file_path: str, sandbox_id: str = None):
        sandbox_id = sandbox_id or self._sandbox_id
        if not sandbox_id:
            logger.error("sandbox_id not set. Cannot download file.")
            return None

        headers = self._get_headers()
        payload = {
            "path": file_path,
        }
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/sandboxes/{sandbox_id}/filesystem/download",
                headers=headers,
                data=json.dumps(payload)
            ) as response:
                if not response.ok:
                    logger.error(f"Download failed: {response.status}")
                    response.raise_for_status()
                return await response.read()  # 返回 bytes
        except Exception:
            logger.exception(f"Failed to download file {file_path}")
            return None

    async def alist_code_sandbox_dir_file_paths(self, file_dir_path: str, sandbox_id: str = None):
        sandbox_id = sandbox_id or self._sandbox_id
        if not sandbox_id:
            logger.error("sandbox_id not set. Cannot list dir files.")
            return json.dumps({"paths": ""}, ensure_ascii=False)

        headers = self._get_headers()
        payload = {
            "path": file_dir_path,
        }

        result = None
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/sandboxes/{sandbox_id}/filesystem",
                headers=headers,
                data=json.dumps(payload)
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

    async def areplace_code_sandbox_image_paths(self, final_result: str, sandbox_id: str = None) -> str:
        sandbox_id = sandbox_id or self._sandbox_id

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

                image_blob = await self.adownload_result_from_sandbox(local_filepath_full, sandbox_id)
                if not isinstance(image_blob, bytes):
                    logger.warning(f"Download did not return bytes for {file_name}")
                    continue

                image_file = BytesIO(image_blob)

                if ext != '.svg' :
                    image_file = compress_image_if_needed(image_file)
                    if not image_file:
                        logger.warning(f"Image compression failed for {file_name}, skipping.")
                        continue

                destination_file_path = f"default_chat_docs/docs/{file_name}"
                await file_store.write_async(
                    file=image_file,
                    file_name=file_name,
                    file_path=destination_file_path,
                    tenant_id=self.tenant_id
                )
                url = await file_store.get_url_async(file_path=destination_file_path, tenant_id=self.tenant_id)
                replacement = f"{prefix}{url}{suffix}"
                replacements.append((match.start(), match.end(), replacement))

            except Exception as e:
                logger.error(f"Failed to process image {file_name}: {e}")
                continue

        result = final_result
        for start, end, repl in reversed(replacements):
            result = result[:start] + repl + result[end:]

        return result

    async def aexecute(self, code: str, timeout: Optional[int] = None, sandbox_id: str = None, context_id: str = None) -> str:
        code = self._extract_code(code)
        await self._ensure_sandbox_initialized()
        sandbox_id = sandbox_id or self._sandbox_id
        context_id = context_id or self._sandbox_context_id
        if not code:
            logger.error("Empty or invalid code provided")
            raise CodeSandboxEmptyCodeException("Empty or invalid code provided")
        if not sandbox_id or not context_id:
            logger.error("sandbox or context not initialized")
            raise CodeSandboxNotInitializedException("sandbox or context not initialized")

        headers = self._get_headers()

        payload = {"code": code, "timeout": timeout or self.timeout_default}
        url = f"{self.base_url}/sandboxes/{sandbox_id}/contexts/{context_id}/execute"
        result = None
        response_text = None
        # 使用共享 session
        session = await HttpSessionShared.ensure_session()
        try:
            async with session.post(
                url,
                headers=headers,
                json=payload
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Execute failed: {response.status} {text}")
                    raise CodeSandboxHTTPException(f"HTTP {response.status}: {text}")

                # Try to get response text first for debugging
                response_text = await response.text()
                logger.debug(f"API response text: {response_text}")

                # Try to parse JSON
                try:
                    result = json.loads(response_text) if response_text else None
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {response_text[:500] if response_text else 'Empty response'}")
                    raise CodeSandboxAPIException(f"Invalid JSON response from API: {str(e)}")

        except asyncio.TimeoutError:
            logger.error("Execution timed out")
            raise CodeSandboxTimeoutException("Execution timed out")
        except (CodeSandboxHTTPException, CodeSandboxTimeoutException, CodeSandboxAPIException):
            raise
        except Exception as e:
            logger.exception("Error during code execution")
            raise CodeSandboxExecutionException(f"Error during code execution: {str(e)}") from e

        results = result.get("results", [])
        stdout_lines = []
        stderr_lines = []
        error_lines = []
        timed_out = False
        result_lines = []
        for item in results:
            typ = item.get("type")
            if typ == "stdout":
                stdout_lines.append(item.get("text", ""))
            elif typ == "result":
                result_lines.append(item.get("text", ""))
            elif typ == "stderr":
                stderr_lines.append(item.get("text", ""))
            elif typ == "error":
                error_lines.append(f"{item.get('name', '')}: {item.get('value', '')}")
            elif typ == "timeout":
                timed_out = True

        parts = []
        if stdout_lines:
            parts.append("stdout:\n" + "".join(stdout_lines).rstrip())
        if result_lines:
            parts.append("result:\n" + "".join(result_lines).rstrip())
        if stderr_lines:
            parts.append("stderr:\n" + "".join(stderr_lines).rstrip())
        if error_lines:
            parts.append("error:\n" + "\n".join(error_lines).rstrip())
        if timed_out or any("TimeoutError" in line for line in stderr_lines):
            logger.error("Execution timed out in results")
            raise CodeSandboxTimeoutException("Execution timed out")

        final_result = "\n".join(parts).strip()
        final_result = await self.areplace_code_sandbox_image_paths(final_result, sandbox_id)
        return final_result if final_result else "Finished execution, but no result."

    async def aexecute_command(self, command: str, cwd: Optional[str] = None, sandbox_id: str = None) -> dict:
        """
        通过终端 cmd 端点同步执行命令

        Args:
            command: 要执行的命令
            cwd: 可选的工作目录
            sandbox_id: 可选的 sandbox ID，如果不提供则使用当前实例的 sandbox_id

        Returns:
            包含执行结果的字典，格式：
            {
                "executionId": str,
                "status": str,
                "result": {
                    "exitCode": int,
                    "stdout": str,
                    "stderr": str,
                    "cwd": str,
                    "executionTimeMs": int
                },
                "executionTimeMs": int
            }
        """
        await self._ensure_sandbox_initialized()
        sandbox_id = sandbox_id or self._sandbox_id
        if not sandbox_id:
            logger.error("sandbox not initialized")
            raise CodeSandboxNotInitializedException("sandbox not initialized")

        headers = self._get_headers()

        payload = {
            "command": command,
        }
        if cwd:
            payload["cwd"] = cwd

        url = f"{self.base_url}/sandboxes/{sandbox_id}/processes/cmd"
        session = await self._get_session()

        try:
            async with session.post(
                url,
                headers=headers,
                json=payload
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Command execution failed: {response.status} {text}")
                    raise CodeSandboxHTTPException(f"HTTP {response.status}: {text}")

                response_text = await response.text()
                logger.debug(f"Command execution response: {response_text}")

                try:
                    result = json.loads(response_text) if response_text else None
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {response_text[:500] if response_text else 'Empty response'}")
                    raise CodeSandboxAPIException(f"Invalid JSON response from API: {str(e)}")

                return result

        except asyncio.TimeoutError:
            logger.error("Command execution timed out")
            raise CodeSandboxTimeoutException("Command execution timed out")
        except (CodeSandboxHTTPException, CodeSandboxTimeoutException, CodeSandboxAPIException):
            raise
        except Exception as e:
            logger.exception("Error during command execution")
            raise CodeSandboxExecutionException(f"Error during command execution: {str(e)}") from e

    async def ainstall_package(self, package_name: str, sandbox_id: str = None, cwd: Optional[str] = None) -> dict:
        """
        使用 sudo pip install 安装 Python 包

        Args:
            package_name: 要安装的包名（可以是单个包名或包含版本号的包名，如 "numpy" 或 "numpy==1.21.0"）
            sandbox_id: 可选的 sandbox ID，如果不提供则使用当前实例的 sandbox_id
            cwd: 可选的工作目录

        Returns:
            包含安装结果的字典，格式与 aexecute_command 相同
        """
        command = f"sudo pip install {package_name}"
        logger.info(f"Installing package: {package_name}")
        result = await self.aexecute_command(command, cwd=cwd, sandbox_id=sandbox_id)

        # 检查执行结果
        if result.get("status") == "completed":
            exit_code = result.get("result", {}).get("exitCode", -1)
            if exit_code == 0:
                logger.info(f"Package {package_name} installed successfully")
            else:
                stdout = result.get("result", {}).get("stdout", "")
                stderr = result.get("result", {}).get("stderr", "")
                logger.warning(f"Package installation exited with code {exit_code}. stdout: {stdout}, stderr: {stderr}")
        else:
            logger.warning(f"Package installation status: {result.get('status')}")

        return result

    async def aget_list_directory_files_tool(self, sandbox_id: str = None):
        async def _wrapped_list_files(file_dir_path: str, sandbox_id: str = None):
            sandbox_id = sandbox_id or self._sandbox_id
            return await self.alist_code_sandbox_dir_file_paths(file_dir_path, sandbox_id)

        description = f"""
        列出 CodeSandbox 中指定目录（如 {DEFAULT_CODE_SANDBOX_DIR_PATH}）下的所有文件路径。
        会自动过滤掉 {DEFAULT_CODE_SANDBOX_SYSTEM_FILES} 等系统文件。
        输入应为一个目录路径字符串。
        """

        return FunctionTool.from_defaults(
            async_fn=_wrapped_list_files,
            name="list-sandbox-files",
            description=description,
            return_direct=False,
        )

    async def aupload_files_to_code_sandbox(self, file_ids: List[str], sandbox_id: str = None):
        sandbox_id = sandbox_id or self._sandbox_id
        if not sandbox_id:
            logger.error("sandbox ID not set. Cannot upload files to sandbox.")
            return None
        for file_id in file_ids:

            file_entity = await self.file_service.get_file(file_id=file_id, tenant_id=self.tenant_id)
            if not file_entity:
                logger.error(f"File entity not found for file ID: {file_id}")
                continue
            file_content_bytes = await file_store.read_async(file_path=file_entity.file_path, tenant_id=self.tenant_id)
            await self.aupload_data_file_to_sandbox(file_content=file_content_bytes, file_name=file_entity.file_name, sandbox_id=sandbox_id)
        logger.info(f"{len(file_ids)} files uploaded to sandbox successfully.")
