# config/providers/codesandbox_provider.py
from typing import Type, Any
from sqlmodel import SQLModel
from pydantic import Field
from config.providers.base_provider import BaseConfigProvider
from db.models.code_sandbox import CodeSandboxConfigEntity
from chat.tools.code_tool import CodeSandboxTool
from llama_index.core.tools import FunctionTool
from loguru import logger
import traceback
import asyncio

# 全局单例
codesandbox_provider: "CodeSandboxProvider"


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

    def get_code_sandbox_tool(self, code_sandbox_ready: asyncio.Future):
        async def aexecute_code(
            code: str,
            timeout: int = 50,
        ) -> str:
            if self.tool is None:
                return "[CodeSandbox Error]: Not configured."
            try:
                await code_sandbox_ready
                return await self.tool.aexecute(code, timeout=timeout)
            except Exception as e:
                logger.error(f"CodeSandbox execution failed: {e}")
                return f"[CodeSandbox Error]: {str(e)}"

        tool = FunctionTool.from_defaults(
        async_fn=aexecute_code,
        name="PythonInterpreter",
        description="""Execute Python code via a secure REST API and return the execution output.
            Params:
            - code: required, string, the Python code to execute.
            - To return any text-based result (e.g., numbers, strings, lists, DataFrame previews), you **must use `print()`**.
                - For DataFrames, always use `print(df.head())` (or `print(df.head(N))`) to show a preview of the data. Do not rely on implicit display.
            - For visualizations (e.g., Matplotlib, Seaborn):
                - **Do not rely on `plt.show()`** — it does not produce any output in this environment.
                - **You must explicitly save the plot to a file** using `plt.savefig('filename.png')` with a descriptive filename** that reflects the chart’s content, such as:
                    - `sales_by_channel_aug2024.png`
                    - `user_growth_q3.png`
                    - `temperature_vs_energy_consumption.png`
                    Avoid generic names like `chart.png`, `plot.png`, or `output.png`.
                - - **To display the image in the response, print it as a Markdown image** using the same descriptive filename: (e.g., `print("Plot saved to: ![](sales_by_channel_aug2024.png)")`).
            - Supported code formats: raw string, wrapped in triple backticks (python ... ), or wrapped in <code>...</code> XML tags.
            - timeout: optional, integer, the maximum execution time in-seconds (default: 50). Must be a positive integer.

            Returns:
            - A string containing the execution result, formatted as follows:
            - If the code prints output, it appears as:
                `"stdout:\n<printed text>"`
            - If runtime errors occur, they appear as:
                `"stderr:\n<error traceback>"`
            - If timeout, they appear as:
                `"timeout: timeout"`
            - If the code runs successfully but prints nothing:
                `"finish: Finished execution."`

            - **Note**:
                - `"type"`: one of `"stdout"`, `"stderr"`, `"timeout"`, etc.
                - `"text"`: the actual content (for `"stdout"`/`"stderr"` types).
            """,
        )
        return tool


# 初始化全局实例
codesandbox_provider = CodeSandboxProvider()
