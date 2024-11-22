import os
import json

from pai_rag.evaluation.dataset.rag_eval_dataset import (
    EvaluationSample,
    PaiRagEvalDataset,
)
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
)
from loguru import logger
from pai.llm_eval.evals.default_templates import DefaultPromptTemplateCN
from pai.llm_eval.pipeline.pipeline_utils import run_rag_offline_eval_pipeline
from pai_rag.evaluation.evaluator.base_evaluator import BaseEvaluator


class PaiEvaluator(BaseEvaluator):
    def __init__(self, llm_config, persist_path: str = None):
        self._llm_config = llm_config
        self.persist_path = persist_path

        self.retrieval_evaluators = [
            DefaultPromptTemplateCN.RETRIEVER_RELEVANCE_PROMPT_TEMPLATE
        ]

        self.response_evaluators = [
            DefaultPromptTemplateCN.FAITHFULNESS_PROMPT_TEMPLATE,
            DefaultPromptTemplateCN.RAG_CORRECTNESS_PROMPT_TEMPLATE,
        ]

        self.created_by = CreatedBy(
            type=CreatedByType.AI, model_name=self._llm_config["model_name"]
        )
        self.qca_dataset_path = os.path.join(self.persist_path, "qca_dataset.json")
        self.evaluation_dataset_path = os.path.join(
            self.persist_path, "evaluation_dataset.json"
        )
        self._show_progress = True
        self._workers = 2

    def format_retrieval_result(self, examples, results):
        ret_examples = []
        for idx, qca_sample in enumerate(examples):
            res = json.loads(results[idx])
            retrieval_eval_example = EvaluationSample(**vars(qca_sample))
            setattr(
                retrieval_eval_example,
                "hitrate",
                res["eval_results"]["0"]["rag"]["retriever_relevance"]["hit_rate"],
            )
            setattr(
                retrieval_eval_example,
                "mrr",
                res["eval_results"]["0"]["rag"]["retriever_relevance"]["mrr"],
            )
            ret_examples.append(retrieval_eval_example)
        return ret_examples

    def format_response_result(self, examples, results):
        ret_examples = []
        for idx, qca_sample in enumerate(examples):
            res = json.loads(results[idx])
            retrieval_eval_example = EvaluationSample(**vars(qca_sample))
            setattr(retrieval_eval_example, "evaluated_by", self.created_by)
            setattr(
                retrieval_eval_example,
                "faithfulness_score",
                res["eval_results"]["0"]["rag"]["faithfulness"]["score"][0],
            )
            setattr(
                retrieval_eval_example,
                "correctness_score",
                res["eval_results"]["0"]["rag"]["rag_correctness"]["score"][0],
            )
            ret_examples.append(retrieval_eval_example)
        return ret_examples

    async def aevaluation(self, stage):
        """Run evaluation with qca dataset."""
        _status = {"retrieval": False, "response": False}
        evaluation_dataset = self.load_evaluation_dataset()
        qca_dataset = self.load_qca_dataset()
        if evaluation_dataset:
            logger.info(
                f"A evaluation dataset already exists with status: [[{evaluation_dataset.status}]]"
            )
            _status = evaluation_dataset.status
            if _status[stage]:
                return evaluation_dataset.results[stage]
            else:
                qca_dataset = evaluation_dataset
        if qca_dataset:
            logger.info(
                f"Starting to generate evaluation dataset for stage: [[{stage}]]..."
            )
            if stage == "retrieval":
                retrieval_result = run_rag_offline_eval_pipeline(
                    self.retrieval_evaluators,
                    self.qca_dataset_path,
                    eval_name="test_rag_offline_eval_retrieval",
                    batch_size=1,
                    need_data_management=False,
                    **self._llm_config,
                )
                eval_examples = self.format_retrieval_result(
                    qca_dataset.examples, retrieval_result
                )
            elif stage == "response":
                response_result = run_rag_offline_eval_pipeline(
                    self.response_evaluators,
                    self.qca_dataset_path,
                    eval_name="test_rag_offline_eval_response",
                    batch_size=1,
                    need_data_management=False,
                    **self._llm_config,
                )
                eval_examples = self.format_response_result(
                    qca_dataset.examples, response_result
                )
            else:
                raise ValueError(f"Invalid stage: {stage}")
            _status[stage] = True
            eval_dataset = PaiRagEvalDataset(examples=eval_examples, status=_status)
            eval_dataset.save_json(self.evaluation_dataset_path)
            return eval_dataset.results[stage]
        else:
            raise ValueError(
                "No QCA dataset found. Please provide a QCA dataset or "
                "generate one first."
            )
