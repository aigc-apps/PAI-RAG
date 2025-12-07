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
from typing import List
from utils.lru_cache import LruCache
from service.knowledgebase.file_service import FileService

search_cache = LruCache(max_size=10)
chatdb_cache = LruCache(max_size=10)


def create_search_tools(websearch_config: WebSearchConfigEntity) -> List[FunctionTool]:
    search_key = json.dumps(websearch_config.model_dump(), ensure_ascii=False, sort_keys=True)
    search_tool = search_cache.get(search_key)
    if search_tool:
        return [search_tool]

    searcher_type = websearch_config.type
    if searcher_type == "tavily":
        search_tool = TavilySearchTool(
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
            raise ValueError("搜索尚未配置.")

        res = await search_client.aquery(query)
        return json.dumps(res, ensure_ascii=False)

    if websearch_config.type == "tavily":
        search_tool = FunctionTool.from_defaults(
            async_fn=aget_search_result,
            name="tavily-websearch",
            description="从 Tavily 搜索引擎中搜索给定查询的最新内容。",
        )
    else:
        search_tool = FunctionTool.from_defaults(
            async_fn=aget_search_result,
            name="aliyun-websearch",
            description="从阿里云搜索引擎中搜索给定查询的最新内容。",
        )

    search_cache.put(search_key, search_tool)
    return [search_tool]


def create_chatdb_tools(chatdb_config: ChatDbConfigEntity) -> List[FunctionTool]:
    chatdb_key = json.dumps(chatdb_config.model_dump(), ensure_ascii=False, sort_keys=True)
    chatdb_tool = chatdb_cache.get(chatdb_key)
    if chatdb_tool:
        return [chatdb_tool]

    chatdb_client = XiyanClient(
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
        )
    chatdb_cache.put(chatdb_key, chatdb_tool)
    return [chatdb_tool]


def create_codesandbox_tools(codesandbox_config: CodeSandboxConfigEntity, code_sandbox_attachments_ids: list[str] = None, file_service: FileService = None):
    code_tool = CodeSandboxTool(
        aliyun_id=codesandbox_config.aliyun_id,
        interpreter_id=codesandbox_config.interpreter_id,
        timeout_default=codesandbox_config.timeout_default,
        enabled=codesandbox_config.enabled,
        code_sandbox_attachments_ids=code_sandbox_attachments_ids,
        file_service=file_service,
    )
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

    tool = FunctionTool.from_defaults(
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
    )
    return tool
