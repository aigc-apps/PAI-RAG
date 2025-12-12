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

    def __init__(self):
        super().__init__()
        self._current_code_sandbox_tool_instance: Any = None

    def _refresh(self, code_sandbox_entity: CodeSandboxConfigEntity):
        logger.info("🔄 _refresh called! Reinitializing CodeSandboxTool.")
        try:
            self.tool_config = CodeSandboxConfig(
                aliyun_id=code_sandbox_entity.aliyun_id,
                interpreter_id=code_sandbox_entity.interpreter_id,
                interpreter_name=code_sandbox_entity.interpreter_name,
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

    def _get_or_create_code_tool(self, code_sandbox_attachments_ids: list = None):
        """
        获取或创建 CodeSandboxTool 实例
        如果已存在实例，则复用；否则创建新实例
        """
        if self._current_code_sandbox_tool_instance is None:
            code_tool = CodeSandboxTool(
                aliyun_id=self.tool_config.aliyun_id,
                interpreter_id=self.tool_config.interpreter_id,
                interpreter_name=self.tool_config.interpreter_name,
                timeout_default=self.tool_config.timeout_default,
                enabled=self.tool_config.enabled,
                code_sandbox_attachments_ids=code_sandbox_attachments_ids,
            )
            # 保存当前实例引用，以便后续清理
            self._current_code_sandbox_tool_instance = code_tool
            logger.info("Created new CodeSandboxTool instance")
        else:
            code_tool = self._current_code_sandbox_tool_instance
            logger.info("Reusing existing CodeSandboxTool instance")
        return code_tool

    def get_code_sandbox_tool(self, code_sandbox_attachments_ids: list = None):
        code_tool = self._get_or_create_code_tool(code_sandbox_attachments_ids)
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

    def get_install_package_tool(self, code_sandbox_attachments_ids: list = None):
        """获取安装 Python 包的工具"""
        code_tool = self._get_or_create_code_tool(code_sandbox_attachments_ids)

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

        tool = FunctionTool.from_defaults(
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
        )
        return tool

    def get_current_code_sandbox_tool_instance(self):
        """获取当前活动的 CodeSandboxTool 实例"""
        return self._current_code_sandbox_tool_instance

    async def aclear_current_code_sandbox_tool_instance(self):
        """清除当前 CodeSandboxTool 实例引用并删除 sandbox 实例"""
        if self._current_code_sandbox_tool_instance:
            try:
                await self._current_code_sandbox_tool_instance.adelete_sandbox_instance()
            except Exception as e:
                logger.exception(f"Failed to delete sandbox instance: {e}")
            finally:
                self._current_code_sandbox_tool_instance = None


# 初始化全局实例
codesandbox_provider = CodeSandboxProvider()
