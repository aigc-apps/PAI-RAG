from llama_index.core.tools import FunctionTool
from db.models.websearch import WebSearchConfigEntity
from db.models.chatdb.chatdb import ChatDbConfigEntity
from db.models.code_sandbox import CodeSandboxConfigEntity
from common.encrypt_utils import decrypt_key
from tools.chatdb.xiyan_client import XiyanClient
from tools.search.aliyun_search import AliyunSearchTool
from tools.search.tavily_search import TavilySearchTool
from tools.code.code_sandbox_tool import CodeSandboxTool
from tools.code.code_sandbox_exceptions import CodeSandboxNotConfiguredException, CodeSandboxException
import json
from loguru import logger
from typing import List, Callable
from utils.lru_cache import LruCache
from service.file.file_resource_service import FileResourceService
from common.llm.llm_model import PaiLlm

search_cache = LruCache(max_size=10)
chatdb_cache = LruCache(max_size=10)


def create_search_tools(websearch_config: WebSearchConfigEntity) -> List[FunctionTool]:
    search_key = json.dumps(websearch_config.model_dump(), ensure_ascii=False, sort_keys=True)
    search_tool = search_cache.get(search_key)
    if search_tool:
        return [search_tool]

    searcher_type = websearch_config.type
    if searcher_type == "tavily":
        search_client = TavilySearchTool(
            api_key=decrypt_key(websearch_config.encrypted_tavily_api_key),
            search_count=websearch_config.search_count,
        )
    elif searcher_type == "aliyun":
        search_client = AliyunSearchTool(
            access_key_id=decrypt_key(websearch_config.encrypted_access_key_id),
            access_key_secret=decrypt_key(websearch_config.encrypted_access_key_secret),
            endpoint=websearch_config.endpoint,
            search_count=websearch_config.search_count,
        )
    else:
        raise ValueError(f"Unsupported searcher type: {searcher_type}")

    async def aget_search_result(query: str) -> str:
        if search_client is None:
            raise ValueError("Search not configured.")

        res = await search_client.aquery(query)
        return json.dumps(res, ensure_ascii=False)

    websearch_description = """Search the web for up-to-date information and return relevant results.

# When to use
You MUST call this tool whenever the user's query involves ANY of the following:
- Current events, news, or recent happenings
- Real-time data: weather, stock prices, sports scores, exchange rates
- Facts that may have changed: population, rankings, product info, policies
- People, companies, or organizations (latest status, announcements)
- Technical questions: APIs, libraries, error messages, best practices
- Any topic you are not 100% certain about
- Any follow-up question on a new topic, even within a multi-turn conversation

# When NOT to use
Only skip this tool for simple greetings ("hi"), trivial math ("1+1"), or universally known constants ("speed of light").

# Parameters
- **query** (required, string): A clear, specific search query. For time-sensitive topics, include the date or time period (e.g. "2026年6月 xxx"). Rewrite vague references into standalone queries.

# Returns
- A JSON object containing a list of search results with titles, URLs, and content snippets.
"""

    if websearch_config.type == "tavily":
        search_tool = FunctionTool.from_defaults(
            async_fn=aget_search_result,
            name="tavily-websearch",
            description=websearch_description,
            return_direct=False,
        )
    else:
        search_tool = FunctionTool.from_defaults(
            async_fn=aget_search_result,
            name="aliyun-websearch",
            description=websearch_description,
            return_direct=False,
        )

    search_cache.put(search_key, search_tool)
    return [search_tool]


def create_chatdb_tools(chatdb_config: ChatDbConfigEntity, chatdb_llm: PaiLlm) -> List[FunctionTool]:
    chatdb_key = f"{chatdb_config.tenant_id}-{chatdb_config.db_name}-{chatdb_config.username}-{chatdb_config.port}-{chatdb_config.host}--{chatdb_config.encrypted_password}"
    chatdb_tool = chatdb_cache.get(chatdb_key)
    if chatdb_tool:
        return [chatdb_tool]

    chatdb_client = XiyanClient(
            llm=chatdb_llm,
            dialect=chatdb_config.dialect,
            host=chatdb_config.host,
            port=chatdb_config.port,
            db_name=chatdb_config.db_name,
            username=chatdb_config.username,
            password=decrypt_key(chatdb_config.encrypted_password),
        )

    chatdb_tool = FunctionTool.from_defaults(
            async_fn=chatdb_client.execute_async,
            name="chat-db",
            description="使用自然语言从给定的数据库中获取数据。输入参数: query(str类型),表示用户的查询意图，需结合上下文信息生成。",
            return_direct=False,
        )
    chatdb_cache.put(chatdb_key, chatdb_tool)
    return [chatdb_tool]


def create_codesandbox_tools(
    codesandbox_config: CodeSandboxConfigEntity,
    code_sandbox_attachments_ids: list[str] = None,
    file_service: FileResourceService = None,
    tenant_id: str = None,
) -> tuple[List[FunctionTool], Callable]:
    """
    创建 CodeSandbox 工具列表
    每次调用都会创建新的 CodeSandboxTool 实例，确保每次 chat 会话使用独立的实例
    返回执行代码工具和安装包工具，它们共享同一个 code_tool 实例
    同时返回清理函数用于清理 sandbox 实例

    Returns:
        tuple: (工具列表, 清理函数)
    """
    code_tool = CodeSandboxTool(
        aliyun_id=codesandbox_config.aliyun_id,
        interpreter_id=codesandbox_config.interpreter_id,
        interpreter_name=codesandbox_config.interpreter_name,
        timeout_default=codesandbox_config.timeout_default or 50,
        enabled=codesandbox_config.enabled,
        code_sandbox_attachments_ids=code_sandbox_attachments_ids,
        file_service=file_service,
        tenant_id=tenant_id,
        api_key=decrypt_key(codesandbox_config.encrypted_api_key) if codesandbox_config.encrypted_api_key else None,
    )

    # 执行代码的工具
    async def aexecute_code(
        code: str,
    ) -> str:
        if code_tool is None:
            logger.error("CodeSandbox not configured")
            raise CodeSandboxNotConfiguredException("Not configured")
        try:
            return await code_tool.aexecute(code)
        except CodeSandboxException as e:
            logger.error(f"CodeSandbox execution failed: {e}")
            raise
        except Exception as e:
            logger.error(f"CodeSandbox execution failed: {e}")
            raise

    execute_tool = FunctionTool.from_defaults(
        async_fn=aexecute_code,
        name="PythonInterpreter",
        description="""Execute Python code with file system access and return the execution output. Use this tool **only** for complex math, spreadsheet analysis, or data visualization.

                # Parameters
                    **IMPORTANT: You MUST pass parameters as a valid JSON object in the format: `{"code": "your_python_code_here"}`**

                    - **`code`** (required, string): The Python code to execute. Pass this as a JSON object with the key "code".
                        - ✅ ALL OUTPUT MUST BE EXPLICITLY PRINTED USING print()
                            - This includes numbers, strings, lists, dictionaries, DataFrames, model metrics, file paths, or any result you want to see.
                            - For DataFrames, always inspect with print(df.head()), print(df.shape), print(df.columns), or print(df.info()) — do not rely on automatic display.
                            - For scalar results, always wrap in print(...), e.g., print(correlation), not just correlation.
                        - For visualizations (Matplotlib, Seaborn, etc.):
                            - **Do not use `plt.show()`** — it has no effect.
                            - **Save the plot** with a **descriptive filename**, e.g.: `sales_by_channel_aug2024.png`, `user_growth_q3.png` (Avoid generic names like `plot.png`.)
                            - **Display the image** by printing a Markdown embed: `print("![Sales by Channel](sales_by_channel_aug2024.png)")`.
                        - Code may be provided as a raw string, in triple backticks (```python ... ```), or in `<code>...</code>` tags. When passing to this tool, wrap it in JSON: `{"code": "your_code_here"}`.


                # Returns
                    - A string containing all printed output from execution, including Markdown image references if plots are generated.
            """,
        return_direct=False,
    )

    # 安装包的工具
    async def ainstall_package(
        package_name: str,
    ) -> str:
        if code_tool is None:
            logger.error("CodeSandbox not configured")
            raise CodeSandboxNotConfiguredException("Not configured")
        try:
            result = await code_tool.ainstall_package(package_name)
        except CodeSandboxException as e:
            logger.error(f"Package installation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Package installation failed: {e}")
            raise
        return result.get("status", "unknown")

    install_tool = FunctionTool.from_defaults(
        async_fn=ainstall_package,
        name="InstallPythonPackage",
        description="""Install a Python package in the CodeSandbox environment using sudo pip install.

                # Parameters
                    **IMPORTANT: You MUST pass parameters as a valid JSON object in the format: `{"package_name": "package_name"}`**

                    - **`package_name`** (required, string): The name of the Python package to install.
                        - Can be a simple package name (e.g., "numpy")
                        - Can include version specification (e.g., "numpy==1.21.0", "pandas>=1.5.0")
                        - Can install multiple packages separated by spaces (e.g., "numpy pandas matplotlib")
                        - Pass this as a JSON object with the key "package_name".

                # Returns
                    - A string containing the installation status, exit code, execution time, stdout, and stderr output.
                    - If installation succeeds (exit code 0), the package is ready to use.
                    - If installation fails, check the stderr output for error details.

                # Usage Examples
                    - Install a single package: `{"package_name": "numpy"}`
                    - Install with version: `{"package_name": "pandas==1.5.0"}`
                    - Install multiple packages: `{"package_name": "numpy scipy matplotlib"}`
            """,
        return_direct=False,
    )

    # 清理函数
    async def cleanup_code_sandbox():
        """清理 CodeSandboxTool 实例"""
        try:
            await code_tool.adelete_sandbox_instance()
        except Exception as e:
            logger.exception(f"Failed to delete sandbox instance: {e}")

    return [execute_tool, install_tool], cleanup_code_sandbox
