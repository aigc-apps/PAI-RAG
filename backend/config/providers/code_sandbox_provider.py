# config/providers/codesandbox_provider.py
from typing import Type, Any
from sqlmodel import SQLModel
from pydantic import Field
from config.providers.base_provider import BaseConfigProvider
from db.models.code_sandbox import CodeSandboxConfigEntity, CodeSandboxConfig
from chat.tools.code_sandbox_tool import CodeSandboxTool
from llama_index.core.tools import FunctionTool
from loguru import logger
import traceback
from chat.tools.code_sandbox_exceptions import (
    CodeSandboxException,
    CodeSandboxNotConfiguredException,
)

# 全局单例
codesandbox_provider: "CodeSandboxProvider"


class CodeSandboxProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = CodeSandboxConfigEntity
    tool_config: Any = Field(default=None)

    def _refresh(self, code_sandbox_entity: CodeSandboxConfigEntity):
        logger.info("🔄 _refresh called! Reinitializing CodeSandboxTool.")
        try:
            self.tool_config = CodeSandboxConfig(
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



    def get_code_sandbox_tool(self, code_sandbox_attachments: list = None):
        code_tool = CodeSandboxTool(
            aliyun_id=self.tool_config.aliyun_id,
            interpreter_id=self.tool_config.interpreter_id,
            timeout_default=self.tool_config.timeout_default,
            enabled=self.tool_config.enabled,
            code_sandbox_attachments=code_sandbox_attachments,
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
            finally:
                # 确保关闭 HTTP session，避免连接泄漏
                try:
                    await code_tool.aclose()
                except Exception as close_error:
                    logger.warning(f"Failed to close CodeSandbox session: {close_error}")

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


# 初始化全局实例
codesandbox_provider = CodeSandboxProvider()
