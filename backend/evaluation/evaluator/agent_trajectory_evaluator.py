import json
from typing import Dict, Any, Optional
import time
from datetime import datetime, timedelta
from evaluation.evaluator.base import BaseEvaluator
from llama_index.core.llms import LLM
from evaluation.evaluator.prompts.correctness import AGENT_TRAJECTORY_PROMPT
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_paillmtrace20240311 import models
from alibabacloud_paillmtrace20240311.client import Client
from pai.llm_eval.common.trace_util import TraceUtil

# 基于arize的AGENT_TRAJECTORY_PROMPT进行评估
class AgentTrajectoryEvaluator(BaseEvaluator):
    """
    基于大语言模型的评估器
    让 LLM 扮演“裁判”角色，对预测结果打分或给出评语
    """

    def __init__(
        self,
        llm: LLM,
        name: str = "AgentTrajectory",
        prompt_template: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        region: str = "cn-hangzhou",
        ak: str = None,
        sk: str = None,
    ):
        super().__init__(name)
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.region = region
        self.ak = ak
        self.sk = sk


    def _default_prompt(self) -> str:
        return AGENT_TRAJECTORY_PROMPT

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM，返回原始响应文本"""
        try:
            response = await self.llm.acomplete(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {str(e)}")

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        import json
        import re
        try:
            pattern = r'(?s)\{.*?\}'
            match = re.search(pattern, response_text)

            if not match:
                raise ValueError("未找到 JSON 对象")

            json_str = match.group(0)
            result = json.loads(json_str)

            if not all(key in result for key in ["score", "reason", "correctness_issues"]):
                raise ValueError("JSON 缺少必要字段")

            return {
                "score": result.get("score", 0.0),
                "reason": result.get("reason", 0.0),
                "evaluator": self.name,
            }

        except json.JSONDecodeError as e:
            return {
                    "score": 0.0,
                    "reason": f"JSON 解析失败: {str(e)}\n原始文本: {response_text[:200]}...",
                    "evaluator": self.name,
                    "error": str(e),
                }
        except Exception as e:
            return {
                    "score": 0.0,
                    "reason": f"提取失败: {str(e)}",
                    "evaluator": self.name,
                    "error": str(e),
                }



    async def evaluate_async(self, input:str, prediction: str, reference: str, trace_id: str, **kwargs) -> Dict[str, Any]:
        del prediction, reference
        if trace_id:
            tool_calls, tools = self._get_tool_calls(trace_id)
            tool_calls = json.dumps(tool_calls, ensure_ascii=False)
            tools = json.dumps(tools, ensure_ascii=False)
            prompt = self.prompt_template.format(
                    inputs=input,
                    tool_calls=tool_calls,
                    tools=tools,
                )
            llm_response = await self._call_llm(prompt)
            result = self._parse_response(llm_response)
        else:
            result = {
                "score": 0.0,
                "reason": "Missing trace_id, tracing must be enabled in AgentTrajectory evaluation.",
                "evaluator": self.name,
            }
        return result

    def _get_tool_calls(self, trace_id):
        """get tool_calls and tools from trace"""
        config = open_api_models.Config(
            access_key_id=self.ak,
            access_key_secret=self.sk,
            protocol='HTTPS',
            region_id=self.region,
            endpoint=f'paillmtrace.{self.region}.aliyuncs.com')
        client = Client(config)

        yesterday = datetime.now().date() - timedelta(days=1)
        request = models.ListTracesDatasRequest(
            min_time=yesterday.strftime('%Y-%m-%d'),
            llm_app_name='',
            page_number=1,
            page_size=1,
            trace_ids=[trace_id]
        )
        for _ in range(3):
            time.sleep(20)
            try:
                resp = client.list_traces_datas(request)
                if resp.body.traces:
                    tools = TraceUtil.get_tools(resp.body.traces[0])
                    tool_calls = TraceUtil.get_tool_calls(resp.body.traces[0])
                    return tool_calls, tools
            except Exception:
                pass

        return [], []
