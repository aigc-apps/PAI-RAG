# tools/codesandbox/code_sandbox_tool.py
import json
import asyncio
import requests
import json5
from typing import Optional, Union
import re
import os
from concurrent.futures import ThreadPoolExecutor
from pairag.file.utils.image_utils import compress_image_if_needed
from utils.tool_utils import aget_file_url_from_db
from llama_index.core.tools import FunctionTool
import uuid
from io import BytesIO
from loguru import logger

_EXECUTOR = ThreadPoolExecutor(max_workers=10)

HEADERS = {"Content-Type": "application/json"}

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
        self.session_id = None
        self.context_id = None

    def create_session_and_context(self):
        s = str(uuid.uuid4())
        session_id = "a" + s[1:]
        headers = {
        "Content-Type": "application/json"
        }
        headers_with_session = headers.copy()
        headers_with_session["X-AgentRun-Session-ID"] = session_id

        payload = {
            "name": "data-analysis",
            "language": "python",
            "config": {}
        }

        response = requests.post(
            f"{self.base_url}/contexts",
            headers=headers_with_session,
            data=json.dumps(payload)
        )

        result = response.json()
        self.session_id = session_id
        self.context_id = result["data"]["id"]
        return session_id, result["data"]["id"]


    def cleanup_session_and_context(self):
        headers = {
        "Content-Type": "application/json",
        }
        headers_with_session = headers.copy()
        headers_with_session["X-AgentRun-Session-ID"] = self.session_id

        # 删除上下文
        requests.delete(
            f"{self.base_url}/contexts/{self.context_id}",
            headers=headers_with_session
        )
        self.session_id = None
        self.context_id = None

    def upload_data_file_to_sandbox(self, file_content, file_name):
        if not self.session_id:
            logger.error("Session ID not set. Cannot upload file to sandbox.")
            return None
        headers_with_session = {}
        headers_with_session["X-AgentRun-Session-ID"] = self.session_id

        try:
            files = {'file': file_content}
            path= os.path.join(DEFAULT_CODE_SANDBOX_DIR_PATH, file_name)
            response = requests.post(
                f"{self.base_url}/files",
                headers=headers_with_session,
                files=files,
                params={'path': path}
            )

            return response.json()
        except Exception as e:
            logger.error(f"Failed to upload file {file_name} to sandbox: {e}")
            return None


    def download_result_from_sandbox(self, file_path):
        headers_with_session = {
            "X-AgentRun-Session-ID": self.session_id,
        }

        try:
            response = requests.get(
                f"{self.base_url}/files",
                headers=headers_with_session,
                params={'path': file_path})
            return response.content
        except Exception as e:
            logger.error(f"Failed to download file {file_path}: {e}")
            return None

    def list_code_sandbox_dir_file_paths(self, file_dir_path):
        if not self.session_id:
            logger.error("Session ID not set. Cannot list dir files.")
            return None
        headers_with_session = {
            "X-AgentRun-Session-ID": self.session_id,
        }

        response = requests.get(
            f"{self.base_url}/filesystem",
            headers=headers_with_session,
            params={'path': file_dir_path}
        )
        result = response.json()
        sandbox_file_paths = []
        if result and 'data' in result and 'entries' in result["data"]:
            sandbox_file_paths = [item['path'] for item in result["data"]['entries'] if item['name'] not in ('.bash_logout', '.bashrc', '.profile')]
            sandbox_file_paths = ','.join(sandbox_file_paths)
        return json.dumps({"data": sandbox_file_paths}, ensure_ascii=False)


    async def alist_code_sandbox_dir_file_paths(self, file_dir_path: str):
        """异步包装同步方法"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _EXECUTOR,
            self.list_code_sandbox_dir_file_paths,
            file_dir_path
        )

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

    async def replace_code_sandbox_image_paths(
    self, final_result: str
) -> str:
        IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
        ext_pattern = '|'.join(ext[1:] for ext in IMAGE_EXTENSIONS)
        MARKDOWN_IMG_PATTERN = re.compile(rf'(!\[[^\]]*\]\()([^\)]+\.(?:{ext_pattern}))(\))', re.IGNORECASE)

        replacements = []

        loop = asyncio.get_running_loop()

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

                image_blob = await loop.run_in_executor(
                    _EXECUTOR,
                    self.download_result_from_sandbox,
                    local_filepath_full
                )

                if not isinstance(image_blob, bytes):
                    logger.warning(f"download_result did not return bytes for {file_name}")
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
            return "[Python Interpreter Error]: Interpreter ID session_id not set."
        actual_timeout = timeout or self.timeout_default


        headers = HEADERS.copy()
        headers["X-AgentRun-Session-ID"] = self.session_id
        headers["X-Acs-Parent-Id"] = self.aliyun_id

        payload = {"code": code}
        url = f"{self.base_url}/contexts/{self.context_id}/execute"

        def _sync_request():
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=actual_timeout + 5,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                raise Exception(f"HTTP request failed: {e}")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_EXECUTOR, _sync_request)

        # 解析结果
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
        final_result = await self.replace_code_sandbox_image_paths(
            final_result
        )
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
