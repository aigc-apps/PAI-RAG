# config/providers/codesandbox_provider.py
from typing import Type, Any, List
from sqlmodel import SQLModel, select
from pydantic import Field
from config.providers.base_provider import BaseConfigProvider
from db.models.code_sandbox import CodeSandboxConfigEntity
from chat.tools.code_sandbox_tool import CodeSandboxTool
from llama_index.core.tools import FunctionTool
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import with_async_db_session
from db.models.knowledgebase.file import KbFileEntity
from pairag.file.store.file_store_helper import file_store
from loguru import logger
import traceback
from chat.tools.code_sandbox_exceptions import (
    CodeSandboxException,
    CodeSandboxNotConfiguredException,
)

# 全局单例
codesandbox_provider: "CodeSandboxProvider"

@with_async_db_session
async def aupload_files_to_code_sandbox(session: AsyncSession, file_ids: List[str], session_id: str):
    file_res = await session.exec(
        select(KbFileEntity).where(
            KbFileEntity.id.in_(file_ids)
        )
    )
    processed_file_entities = file_res.all()
    files = [(entity.file_path, entity.file_name) for entity in processed_file_entities]
    unique_kb_ids = list({entity.kb_id for entity in processed_file_entities})
    assert len(unique_kb_ids) == 1, "file_ids must be from the same knowledgebase"
    for file_path, file_name in files:
        file_content_bytes = file_store.load(file_path)
        await codesandbox_provider.tool.aupload_data_file_to_sandbox(file_content_bytes, file_name, session_id)


class CodeSandboxProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = CodeSandboxConfigEntity
    tool: Any = Field(default=None)

    def _refresh(self, code_sandbox_entity: CodeSandboxConfigEntity):
        logger.info("🔄 _refresh called! Reinitializing CodeSandboxTool.")
        try:
            self.tool = CodeSandboxTool(
                aliyun_id=code_sandbox_entity.aliyun_id,
                interpreter_id=code_sandbox_entity.interpreter_id,
                timeout_default=code_sandbox_entity.timeout_default or 50,
                enabled=code_sandbox_entity.enabled,
            )
        except Exception:
            logger.error(f"Failed to initialize CodeSandboxTool: {traceback.format_exc()}")
            self.tool = None

    def _load_entries(self, entries):
        super()._load_entries(entries)
        if entries:
            self._refresh(entries[0])

    def add(self, entry):
        super().add(entry)
        self._refresh(entry)

    def update(self, entry):
        super().update(entry)
        self._refresh(entry)

    async def initialize_sandbox_with_attachments(self, code_sandbox_attachments: list = None):
        """初始化 sandbox 并上传附件"""
        if self.tool is None:
            raise Exception("CodeSandbox tool not configured")

        # 1. 创建session和context
        session_id, context_id = await self.tool.acreate_session_and_context()

        # 2. 如果有附件，上传文件
        if code_sandbox_attachments:
            logger.info(f"[Model] uploading {len(code_sandbox_attachments)} code sandbox attachments.")
            file_ids = [att["id"] for att in code_sandbox_attachments]
            await aupload_files_to_code_sandbox(file_ids=file_ids, session_id=session_id)
            logger.info("[Model] Code sandbox ready and files uploaded.")
        else:
            logger.info("[Model] Code sandbox ready.")

        return session_id, context_id


    def get_code_sandbox_tool(self):
        async def aexecute_code(
            code: str,
            timeout: int = 50,
            session_id: str = None,
            context_id: str = None,
        ) -> str:
            if self.tool is None:
                logger.error("CodeSandbox not configured")
                raise CodeSandboxNotConfiguredException("Not configured")
            try:
                return await self.tool.aexecute(code, timeout=timeout, session_id=session_id, context_id=context_id)
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
                    - **`code`** (required, string): The Python code to execute.
                        - To return any result (numbers, strings, lists, DataFrames, images), **use `print()`**.
                            - For DataFrames, always use: `print(df.head())`, `print(df.info())`, `print(df.columns)`, etc. Do not rely on implicit display.
                        - For visualizations (Matplotlib, Seaborn, etc.):
                            - **Do not use `plt.show()`** — it has no effect.
                            - **Save the plot** with a **descriptive filename**, e.g.: `sales_by_channel_aug2024.png`, `user_growth_q3.png` (Avoid generic names like `plot.png`.)
                            - **Display the image** by printing a Markdown embed: `print("![Sales by Channel](sales_by_channel_aug2024.png)")`.
                        - Code may be provided as a raw string, in triple backticks (```python ... ```), or in `<code>...</code>` tags.

                    - **`timeout`** (optional, int): Max execution time in seconds (default: 50). Must be positive.
                    - **`session_id`** (optional, string): Sandbox session ID (defaults to current session).
                    - **`context_id`** (optional, string): Sandbox context ID (defaults to current context).

                # Returns
                    - A string containing all printed output from execution, including Markdown image references if plots are generated.
            """,
        )
        return tool


# 初始化全局实例
codesandbox_provider = CodeSandboxProvider()
