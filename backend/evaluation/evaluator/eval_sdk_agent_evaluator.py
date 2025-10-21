# evaluator/eval_sdk_agent_evaluator.py
import asyncio
import json
import time
from datetime import datetime
import uuid
from enum import Enum
from typing import Any, Dict, List
from loguru import logger

from backend.evaluation.evaluator.base import BaseEvaluator

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_paillmtrace20240311 import models
from alibabacloud_paillmtrace20240311.client import Client
from pai.llm_eval.common.eval_constants import EvalModelConfig, ExtraBodyParams, ModelLang
from pai.llm_eval.proto.common_pb2 import AgentEvalData
from pai.llm_eval.pipeline.agent_eval_pipeline import AgentEvalPipeline
from pai.llm_eval.evals.agent_eval_templates import AgentEvalPromptTemplate, AgentEvalPromptTemplateCN



class EvalSdkAgentEvaluator(BaseEvaluator):
    """Use pai llm evals sdk to do various types of agent evaluation tasks,
    such as agent trajectory correctness, agent answer semantic similarity,
    and agent tool call correctness, etc """

    _YMD_HMS_FORMAT = '%Y-%m-%d %H:%M:%S'

    class Task(Enum):
        """The evaluator does these types of evaluations."""
        AGENT_PLAN_VALIDITY = 0
        AGENT_TRAJECTORY_CORRECTNESS = 1
        AGENT_ANSWER_SEMANTIC_SIMILARITY = 2
        AGENT_TOOL_CALL_CORRECTNESS = 3

    class Language(Enum):
        EN_US = 'en-us'
        ZH_CN = 'zh-cn'

    def __init__(self, name: str, task: Task, region: str, tracing_access_key_id: str, tracing_access_key_secret: str,
                 model_name: str, model_endpoint: str, model_api_key: str, temperature: float = 0, top_p: float = 0.99, max_tokens: int = 512,
                 is_self_host: bool = False, use_function_call: bool = False,
                 need_explanation_in_result: bool = True, language: str = Language.EN_US.value):
        if not name:
            name = EvalSdkAgentEvaluator.__name__
        super().__init__(name)

        # model that is used for evaluation
        if not (model_name and model_endpoint and model_api_key):
            raise RuntimeError('EvalSdkAgentEvaluator: missing model name, endpoint or api key')

        if not task:
            raise RuntimeError('EvalSdkAgentEvaluator: missing task input parameter value')

        if not region:
            raise RuntimeError('EvalSdkAgentEvaluator: missing region input parameter value')

        if temperature < 0:
            raise RuntimeError('EvalSdkAgentEvaluator: temperature cannot be negative')

        if top_p < 0 or top_p > 1:
            raise RuntimeError('EvalSdkAgentEvaluator: top_p cannot be negative or greater than 1.0')

        if max_tokens < 0:
            raise RuntimeError('EvalSdkAgentEvaluator: max_tokens cannot be negative')


        tracing_client_config = open_api_models.Config(access_key_id=tracing_access_key_id,
                                                       access_key_secret=tracing_access_key_secret,
                                                       protocol='HTTPS',
                                                       region_id=region,
                                                       endpoint=f'paillmtrace.{region}.aliyuncs.com')
        self._tracing_client = Client(tracing_client_config)



        if not language or language.lower() == EvalSdkAgentEvaluator.Language.ZH_CN.value:
            language = EvalSdkAgentEvaluator.Language.ZH_CN.value
        else:
            language = EvalSdkAgentEvaluator.Language.EN_US.value

        self._model_config = {
            EvalModelConfig.MODEL_NAME: model_name,
            EvalModelConfig.MODEL_API_KEY: model_api_key,
            EvalModelConfig.MODEL_BASE_URL: model_endpoint,
            EvalModelConfig.EXTRA_BODY: {ExtraBodyParams.CHANNEL: 'llm-trace', ExtraBodyParams.LANGUAGE: language},
            EvalModelConfig.TEMPERATURE: temperature,
            EvalModelConfig.TOP_P: top_p,
            EvalModelConfig.MAX_TOKENS: max_tokens,
            EvalModelConfig.IS_SELF_HOST: is_self_host,
            EvalModelConfig.USE_FUNCTION_CALL: use_function_call}

        self._need_explanation_in_result = need_explanation_in_result
        self._eval_prompt_template = EvalSdkAgentEvaluator._get_default_prompt(language, task)

        logger.info(f'initialized EvalSdkAgentEvaluator for task {task} in region {region}, using model {model_name} at {model_endpoint}')


    @staticmethod
    def _get_default_prompt(language:str, task: Task) -> str:
        if language and language.lower() == ModelLang.ZH_CN:
            if task == EvalSdkAgentEvaluator.Task.AGENT_PLAN_VALIDITY:
                return AgentEvalPromptTemplateCN.AGENT_PLAN_VALIDITY_PROMPT_TEMPLATE
            elif task == EvalSdkAgentEvaluator.Task.AGENT_ANSWER_SEMANTIC_SIMILARITY:
                return AgentEvalPromptTemplateCN.AGENT_ANSWER_SEMANTIC_SIMILARITY_PROMPT_TEMPLATE
            elif task == EvalSdkAgentEvaluator.Task.AGENT_TOOL_CALL_CORRECTNESS:
                return AgentEvalPromptTemplateCN.AGENT_TOOL_CALL_CORRECTNESS_PROMPT_TEMPLATE
            else: # EvalSdkAgentEvaluator.Task.AGENT_TRAJECTORY_CORRECTNESS:
                return AgentEvalPromptTemplateCN.AGENT_TRAJECTORY_CORRECTNESS_PROMPT_TEMPLATE
        else:
            if task == EvalSdkAgentEvaluator.Task.AGENT_PLAN_VALIDITY:
                return AgentEvalPromptTemplate.AGENT_PLAN_VALIDITY_PROMPT_TEMPLATE
            elif task == EvalSdkAgentEvaluator.Task.AGENT_ANSWER_SEMANTIC_SIMILARITY:
                return AgentEvalPromptTemplate.AGENT_ANSWER_SEMANTIC_SIMILARITY_PROMPT_TEMPLATE
            elif task == EvalSdkAgentEvaluator.Task.AGENT_TOOL_CALL_CORRECTNESS:
                return AgentEvalPromptTemplate.AGENT_TOOL_CALL_CORRECTNESS_PROMPT_TEMPLATE
            else: # EvalSdkAgentEvaluator.Task.AGENT_TRAJECTORY_CORRECTNESS:
                return AgentEvalPromptTemplate.AGENT_TRAJECTORY_CORRECTNESS_PROMPT_TEMPLATE


    async def evaluate_async(self, input:str, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
        """Implement abstrace method of base class.
        input: input to LLM/Agent system, prediction: response from LLM/Agent system, reference: ground truth"""
        if not input:
            raise RuntimeError('missing input for agent evaluation')

        trace_id = kwargs.get('trace_id', '')
        if not trace_id:
            raise RuntimeError('missing trace_id for agent evaluation')

        trace = await asyncio.to_thread(self._get_trace, trace_id)

        # parse trace
        agent_eval_input = self._build_agent_eval_input(trace_id, trace, input, prediction, reference)
        eval_id = str(uuid.uuid4())
        # dict
        sdk_eval_result = await asyncio.to_thread(AgentEvalPipeline.eval_agent,
                                              input_data=[agent_eval_input],
                                              eval_prompt_template=self._eval_prompt_template,
                                              eval_id=eval_id,
                                              need_explanation=self._need_explanation_in_result,
                                              model_config=self._model_config)

        return self._build_final_eval_result(sdk_eval_result)


    def _get_trace(self, trace_id: str) -> List[Dict]:
        """Call tracing api to get trace data."""
        # time window: [30 min ago, 10 min in the future]
        dt0 = datetime.fromtimestamp(int(time.time()) - 1800)
        dt1 = datetime.fromtimestamp(int(time.time()) + 600)

        trace = []
        try:
            pop_req = models.ListTracesDatasRequest(
                max_time=dt1.strftime(EvalSdkAgentEvaluator._YMD_HMS_FORMAT),
                min_time=dt0.strftime(EvalSdkAgentEvaluator._YMD_HMS_FORMAT),
                page_number=1,
                page_size=1,
                trace_ids=[trace_id],
                trace_reduce_method='REMOVE_EMBEDDING')
            logger.info(f'agent eval, getting trace {trace_id}, pop request:{pop_req.to_map()}')

            pop_resp = self._tracing_client.list_traces_datas(pop_req)
            if pop_resp and pop_resp.body and pop_resp.body.traces:
                # a list of dict, each dict is a span
                trace = json.loads(pop_resp.body.traces[0])
        except Exception as e:
            logger.error(f'agent eval, failed to get trace {trace_id}: {e}')

        return trace


    def _build_agent_eval_input(self, trace_id: str, trace: List[Dict], input: str, prediction: str, ground_truth: str) -> AgentEvalData:
        """Prepare the core input to eval sdk."""
        input = AgentEvalData()
        input.query = input
        # TODO: set all properties
        return input

    def _build_final_eval_result(self, sdk_eval_result: dict) -> Dict[str, Any]:
        """Convert the eval sdk output to standard evaluator output."""
        eval_result = {}
        #TODO: implement
        return eval_result
