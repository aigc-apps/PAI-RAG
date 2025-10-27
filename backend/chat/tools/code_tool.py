# tools/codesandbox/code_sandbox_tool.py
import json
import asyncio
import aiohttp
import json5
from typing import Optional, Union
import re
import os
from pairag.file.utils.image_utils import compress_image_if_needed
from utils.tool_utils import aget_file_url_from_db
from llama_index.core.tools import FunctionTool
import uuid
from io import BytesIO
from aiohttp import FormData
from loguru import logger

DEFAULT_CODE_SANDBOX_DIR_PATH = '/home/user'


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
        self.session_id: Optional[str] = None
        self.context_id: Optional[str] = None
        # 创建一个可复用的 aiohttp ClientSession
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=600)
            )
        return self._http_session

    async def aclose(self):
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    async def acreate_session_and_context(self):
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

        session = await self._get_session()
        url = f"{self.base_url}/contexts"
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

        self.session_id = session_id
        self.context_id = result["data"]["id"]
        return session_id, result["data"]["id"]

    async def acleanup_session_and_context(self):
        if not self.session_id or not self.context_id:
            logger.warning("No active session/context to clean up.")
            return

        headers_with_session = {
            "Content-Type": "application/json",
            "X-AgentRun-Session-ID": self.session_id,
        }

        session = await self._get_session()
        url = f"{self.base_url}/contexts/{self.context_id}"
        try:
            async with session.delete(url, headers=headers_with_session) as response:
                if not response.ok:
                    logger.warning(f"Cleanup failed: {response.status}")
        except Exception:
            logger.exception("Error during cleanup")

        self.session_id = None
        self.context_id = None

    async def aupload_data_file_to_sandbox(self, file_content: bytes, file_name: str):
        if not self.session_id:
            logger.error("Session ID not set. Cannot upload file to sandbox.")
            return None

        headers_with_session = {
            "X-AgentRun-Session-ID": self.session_id,
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

    async def adownload_result_from_sandbox(self, file_path: str):
        if not self.session_id:
            logger.error("Session ID not set. Cannot download file.")
            return None

        headers_with_session = {
            "X-AgentRun-Session-ID": self.session_id,
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

    async def alist_code_sandbox_dir_file_paths(self, file_dir_path: str):
        if not self.session_id:
            logger.error("Session ID not set. Cannot list dir files.")
            return json.dumps({"data": ""}, ensure_ascii=False)

        headers_with_session = {
            "X-AgentRun-Session-ID": self.session_id,
        }

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
            return json.dumps({"data": ""}, ensure_ascii=False)

        sandbox_file_paths = []
        if result and 'data' in result and 'entries' in result["data"]:
            sandbox_file_paths = [
                item['path'] for item in result["data"]['entries']
                if item['name'] not in ('.bash_logout', '.bashrc', '.profile')
            ]
        return json.dumps({"data": ','.join(sandbox_file_paths)}, ensure_ascii=False)

    def _extract_code(self, params: Union[str, dict]) -> str:
        try:
            if isinstance(params, str):
                params = json5.loads(params)
            code = params.get('code', '') or params.get('raw', '')
            triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            if triple_match:
                code = triple_match.group(1)
            else:
                xml_match = re.search(r'<code>(.*?)</code>', code, re.DOTALL)
                if xml_match:
                    code = xml_match.group(1)
            return code.strip()
        except Exception:
            if isinstance(params, str):
                return params.strip()
            return ""

    async def areplace_code_sandbox_image_paths(self, final_result: str) -> str:
        IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
        ext_pattern = '|'.join(ext[1:] for ext in IMAGE_EXTENSIONS)
        MARKDOWN_IMG_PATTERN = re.compile(rf'(!\[[^\]]*\]\()([^\)]+\.(?:{ext_pattern}))(\))', re.IGNORECASE)

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

                image_blob = await self.adownload_result_from_sandbox(local_filepath_full)
                if not isinstance(image_blob, bytes):
                    logger.warning(f"Download did not return bytes for {file_name}")
                    continue

                image_file = BytesIO(image_blob)

                if ext != '.svg':
                    image_file = compress_image_if_needed(image_file)
                    if not image_file:
                        logger.warning(f"Image compression failed for {file_name}, skipping.")
                        continue

                url = await aget_file_url_from_db(file=image_file, file_name=file_name)
                replacement = f"{prefix}{url}{suffix}"
                replacements.append((match.start(), match.end(), replacement))

            except Exception as e:
                logger.error(f"Failed to process image {file_name}: {e}")
                continue

        result = final_result
        for start, end, repl in reversed(replacements):
            result = result[:start] + repl + result[end:]

        return result

    async def aexecute(self, code: str, timeout: Optional[int] = None) -> str:
        code = self._extract_code(code)
        if not code:
            return "[Python Interpreter Error]: Empty or invalid code provided."
        if not self.session_id or not self.context_id:
            return "[Python Interpreter Error]: Session or context not initialized."

        actual_timeout = timeout or self.timeout_default
        headers = {
            "Content-Type": "application/json",
            "X-AgentRun-Session-ID": self.session_id,
            "X-Acs-Parent-Id": self.aliyun_id,
        }

        payload = {"code": code}
        url = f"{self.base_url}/contexts/{self.context_id}/execute"

        session = await self._get_session()
        try:
            # aiohttp 的 timeout 是总超时（包括连接+读取）
            timeout_obj = aiohttp.ClientTimeout(total=actual_timeout + 5)
            async with session.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout_obj
            ) as response:
                if not response.ok:
                    text = await response.text()
                    logger.error(f"Execute failed: {response.status} {text}")
                    raise Exception(f"HTTP {response.status}: {text}")
                result = await response.json()
        except asyncio.TimeoutError:
            return "[PythonInterpreter Error] TimeoutError: Execution timed out."
        except Exception as e:
            logger.exception("Error during code execution")
            return f"[Python Interpreter Error]: {str(e)}"

        if result.get("code") != "SUCCESS":
            return f"[Python Interpreter Error]: API returned non-success code: {result.get('code')}"

        results = result.get("data", {}).get("results", [])
        stdout_lines = []
        stderr_lines = []
        timed_out = False

        for item in results:
            typ = item.get("type")
            if typ == "stdout":
                stdout_lines.append(item.get("text", ""))
            elif typ == "stderr":
                stderr_lines.append(item.get("text", ""))
            elif typ == "timeout":
                timed_out = True

        parts = []
        if stdout_lines:
            parts.append("stdout:\n" + "".join(stdout_lines).rstrip())
        if stderr_lines:
            parts.append("stderr:\n" + "".join(stderr_lines).rstrip())
        if timed_out or any("TimeoutError" in line for line in stderr_lines):
            parts.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")

        final_result = "\n".join(parts).strip()
        final_result = await self.areplace_code_sandbox_image_paths(final_result)
        return final_result if final_result else "Finished execution."

    async def aget_list_directory_files_tool(self):
        async def _wrapped_list_files(file_dir_path: str):
            return await self.alist_code_sandbox_dir_file_paths(file_dir_path)

        return FunctionTool.from_defaults(
            async_fn=_wrapped_list_files,
            name="list-sandbox-files",
            description=(
                "列出 CodeSandbox 中指定目录（如 '/home/user'）下的所有文件路径。"
                "会自动过滤掉 .bashrc、.bash_logout、.profile 等系统文件。"
                "输入应为一个目录路径字符串。"
            ),
        )
