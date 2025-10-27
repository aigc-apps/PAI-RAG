import re
import json5
import os
import random
import asyncio
from typing import Union, Annotated
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.tools import FunctionTool
from loguru import logger

# 假设 sandbox_fusion.run_code 是同步函数（基于 requests）
from sandbox_fusion import run_code, RunCodeRequest, set_sandbox_endpoint
from requests.exceptions import Timeout

# Endpoint 配置
SANDBOX_FUSION_ENDPOINTS = ['http://localhost:8080']
if 'SANDBOX_FUSION_ENDPOINT' in os.environ:
    SANDBOX_FUSION_ENDPOINTS = os.environ['SANDBOX_FUSION_ENDPOINT'].split(',')
set_sandbox_endpoint('http://localhost:8080')

_EXECUTOR = ThreadPoolExecutor(max_workers=10)

async def _run_code_async(code: str, language: str, run_timeout: int, client_timeout: int, endpoint: str):
    print("********start _run_code_async******")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _EXECUTOR,
        run_code,
        RunCodeRequest(code=code, language=language),
    )

def _extract_code(params: Union[str, dict]) -> str:
    try:
        if isinstance(params, str):
            params = json5.loads(params)
        code = params.get('code', '') or params.get('raw', '')
        # 支持 ```code``` 和 <code>...</code>
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

async def apython_interpreter(
    code: Annotated[str, "The Python code to execute. Must be valid Python code. Use print() to output results."] = "",
    timeout: Annotated[int, "Execution timeout in seconds (default: 50)"] = 50,
) -> str:
    if not SANDBOX_FUSION_ENDPOINTS:
        return "[Python Interpreter Error]: No sandbox endpoints configured."

    code = _extract_code(code)
    if not code:
        return "[Python Interpreter Error]: Empty or invalid code provided."

    last_error = None
    max_attempts = 5

    for attempt in range(max_attempts):
        endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)
        logger.info(f"PythonInterpreter attempt {attempt + 1}/{max_attempts} using endpoint: {endpoint}")

        try:
            code_result = await _run_code_async(
                code=code,
                language='python',
                run_timeout=timeout,
                client_timeout=timeout,
                endpoint=endpoint
            )
            print("********finish _run_code_async******")

            parts = []
            if code_result.run_result.stdout:
                parts.append(f"stdout:\n{code_result.run_result.stdout}")
            if code_result.run_result.stderr:
                parts.append(f"stderr:\n{code_result.run_result.stderr}")
            if code_result.run_result.execution_time >= timeout - 1:
                parts.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")

            result = '\n'.join(parts).strip()
            return result if result else "Finished execution."

        except Timeout:
            last_error = f"[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}."
            logger.warning(f"Timeout on attempt {attempt + 1}: {last_error}")
        except Exception as e:
            last_error = f"[Python Interpreter Error]: {str(e)} on endpoint {endpoint}"
            logger.error(f"Error on attempt {attempt + 1}: {last_error}")

        if attempt < max_attempts - 1:
            await asyncio.sleep(0.1)

    return last_error or "[Python Interpreter Error]: All attempts failed."

async def apython_interpreter_tool(code: str, timeout: int = 50) -> str:
    try:
        return await apython_interpreter(code=code, timeout=timeout)
    except Exception as e:
        logger.error(f"Python interpreter tool failed: {e}")
        return f"[Python Interpreter Error]: {str(e)}"

async def aget_python_interpreter_tool():
    """
    Returns an async FunctionTool for executing Python code in a sandbox.
    """
    apython_tool_func = partial(apython_interpreter_tool)

    async def python_handler(
        code: Annotated[
            str,
            "The Python code to execute. Must be valid Python code. Use print() to output results you want to capture.",
        ],
        timeout: Annotated[
            int,
            "Execution timeout in seconds. Default is 50. Must be a positive integer.",
        ] = 50,
        **kwargs
    ) -> str:
        logger.info(f"Executing Python code with timeout={timeout}s")
        return await apython_tool_func(code=code, timeout=timeout)

    tool = FunctionTool.from_defaults(
        async_fn=python_handler,
        name="PythonInterpreter",
        description="""Execute Python code in a secure, sandboxed environment and return the execution output.
Params:
- code: required, string, the Python code to execute. You must use print() statements to output any result you want to see. The code can be provided as raw string, within triple backticks (```python ... ```), or inside <code>...</code> XML tags.
- timeout: optional, integer, the maximum execution time in seconds (default: 50). Must be a positive integer.

Returns:
- A string containing the execution result, which may include:
  - "stdout: ..." if the code produced output via print()
  - "stderr: ..." if there were runtime errors or exceptions
  - "[PythonInterpreter Error] ..." if execution failed, timed out, or no endpoints were available
  - "Finished execution." if the code ran successfully but produced no output

Note: The environment is stateless — variables do not persist across calls.
""",
    )
    return tool
