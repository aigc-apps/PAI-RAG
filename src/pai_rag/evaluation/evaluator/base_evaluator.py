import os
from pai_rag.evaluation.metrics.retrieval.hitrate import HitRate
from pai_rag.evaluation.metrics.retrieval.mrr import MRR
from pai_rag.evaluation.metrics.response.faithfulness import Faithfulness
from pai_rag.evaluation.metrics.response.correctness import Correctness

from llama_index.core.async_utils import run_jobs
from pai_rag.evaluation.dataset.rag_eval_dataset_refactor import (
    QcapEvalSample,
    QcapEvalDataset,
)
from pai_rag.evaluation.dataset.rag_qca_dataset_refactor import QcapDataset
from pai_rag.evaluation.dataset.state_manager import (
    DatasetState,
    StateManager,
)
from loguru import logger


class BaseEvaluator:
    def __init__(
        self,
        llm,
        persist_path: str = None,
        evaluation_dataset_path: str = None,
        enable_multi_modal: bool = False,
        use_granular_metrics: bool = False,
        state_manager: StateManager = None,
    ):
        self._llm = llm
        self.hitrate = HitRate(use_granular_hit_rate=use_granular_metrics)
        self.mrr = MRR(use_granular_mrr=use_granular_metrics)
        self.retrieval_evaluators = [self.hitrate, self.mrr]
        self.faithfulness_evaluator = Faithfulness(
            llm=self._llm,
        )
        self.correctness_evaluator = Correctness(
            llm=self._llm,
        )
        self.response_evaluators = [
            self.faithfulness_evaluator,
            self.correctness_evaluator,
        ]
        self.evaluation_dataset_path = evaluation_dataset_path or os.path.join(
            persist_path, "end2end_evaluation_dataset.jsonl"
        )
        self.retrieval_evaluation_dataset_path = os.path.join(
            persist_path, "retrieval_evaluation_dataset.jsonl"
        )
        self.response_evaluation_dataset_path = os.path.join(
            persist_path, "response_evaluation_dataset.jsonl"
        )
        self.qcap_dataset_path = os.path.join(persist_path, "qcap_dataset.jsonl")
        self.state_manager = state_manager or StateManager(
            os.path.join(persist_path, "state.json")
        )
        self._show_progress = True
        self._workers = 2
        self.enable_multi_modal = enable_multi_modal

    async def compute_retrieval_metrics(self, qcap_sample):
        metrics_results = {}
        reference_node_ids = qcap_sample.qca.get_reference_node_ids()
        predicted_node_ids = qcap_sample.get_predicted_node_ids()
        for metric in self.retrieval_evaluators:
            metric_score = metric.compute(reference_node_ids, predicted_node_ids)
            metrics_results[metric.metric_name] = metric_score
        retrieval_eval_example = QcapEvalSample(
            qcap=qcap_sample,
            eval_results={
                "results": metrics_results,
                "source": {"name": "local", "model": ""},
            },
        )
        return retrieval_eval_example

    async def compute_response_metrics(self, qcap_sample):
        query = qcap_sample.qca.query.query_text
        reference_answer = qcap_sample.qca.answer.answer_text
        response_answer = qcap_sample.prediction.answer.answer_text
        contexts = qcap_sample.qca.get_reference_node_texts()
        reference_image_url_list = qcap_sample.qca.get_reference_image_url_list()
        metrics_results = {}
        for metric in self.response_evaluators:
            if self.enable_multi_modal:
                metric_result = await metric.aevaluate_multimodal(
                    query,
                    reference_answer,
                    contexts,
                    reference_image_url_list,
                    response_answer,
                    sleep_time_in_seconds=3,
                )
            else:
                metric_result = await metric.aevaluate(
                    query,
                    reference_answer,
                    contexts,
                    response_answer,
                    sleep_time_in_seconds=3,
                )

            metrics_results[metric.metric_name] = {
                "score": metric_result[0],
                "reason": metric_result[1],
            }

        response_eval_example = QcapEvalSample(
            qcap=qcap_sample,
            eval_results={
                "results": metrics_results,
                "source": {"name": "AI", "model": self._llm.metadata.model_name},
            },
        )

        return response_eval_example

    async def compute_e2e_metrics(self, qcap_sample):
        reference_node_ids = qcap_sample.qca.get_reference_node_ids()
        predicted_node_ids = qcap_sample.get_predicted_node_ids()
        query = qcap_sample.qca.query.query_text
        reference_answer = qcap_sample.qca.answer.answer_text
        response_answer = qcap_sample.prediction.answer.answer_text
        contexts = qcap_sample.qca.get_reference_node_texts()
        reference_image_url_list = qcap_sample.qca.get_reference_image_url_list()

        metrics_results = {}
        for metric in self.retrieval_evaluators:
            metric_score = metric.compute(reference_node_ids, predicted_node_ids)
            metrics_results[metric.metric_name] = metric_score
        for metric in self.response_evaluators:
            if self.enable_multi_modal:
                metric_result = await metric.aevaluate_multimodal(
                    query,
                    reference_answer,
                    contexts,
                    reference_image_url_list,
                    response_answer,
                    sleep_time_in_seconds=3,
                )
            else:
                metric_result = await metric.aevaluate(
                    query,
                    reference_answer,
                    contexts,
                    response_answer,
                    sleep_time_in_seconds=3,
                )

            metrics_results[metric.metric_name] = {
                "score": metric_result[0],
                "reason": metric_result[1],
            }

        response_eval_example = QcapEvalSample(
            qcap=qcap_sample,
            eval_results={
                "results": metrics_results,
                "source": {"name": "AI", "model": self._llm.metadata.model_name},
            },
        )

        return response_eval_example

    async def aevaluation_for_retrieval(self):
        if self.state_manager.is_completed(DatasetState.RETRIEVAL):
            logger.info("Evaluation dataset for retrieval stage already exists.")
        else:
            logger.info(
                "Starting to generate evaluation dataset for retrieval stage..."
            )
            qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            eval_tasks = []
            for qcap in qcap_dataset.samples:
                eval_tasks.append(self.compute_retrieval_metrics(qcap))
            retrieval_eval_samples = await run_jobs(
                eval_tasks, self._show_progress, self._workers
            )
            retrieval_eval_dataset = QcapEvalDataset(samples=retrieval_eval_samples)
            retrieval_eval_dataset.save_json(self.retrieval_evaluation_dataset_path)
            self.state_manager.mark_completed(DatasetState.RETRIEVAL)
            logger.info("Evaluation dataset for retrieval stage generated.")

    async def aevaluation_for_response(self):
        if self.state_manager.is_completed(DatasetState.RESPONSE):
            logger.info("Evaluation dataset for response stage already exists.")
        else:
            logger.info("Starting to generate evaluation dataset for response stage...")
            qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            eval_tasks = []
            for qcap in qcap_dataset.samples:
                eval_tasks.append(self.compute_response_metrics(qcap))
            response_eval_samples = await run_jobs(
                eval_tasks, self._show_progress, self._workers
            )
            response_eval_dataset = QcapEvalDataset(samples=response_eval_samples)
            response_eval_dataset.save_json(self.response_evaluation_dataset_path)
            self.state_manager.mark_completed(DatasetState.RESPONSE)
            logger.info("Evaluation dataset for response stage generated.")

    async def aevaluation_for_all(self):
        """Run evaluation with qca dataset."""
        if self.state_manager.is_completed(DatasetState.E2E):
            logger.info("Evaluation dataset for end2end stage already exists.")
        else:
            logger.info("Starting to generate evaluation dataset for end2end stage...")
            qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            eval_tasks = []
            for qcap in qcap_dataset.samples:
                eval_tasks.append(self.compute_e2e_metrics(qcap))
            e2e_eval_samples = await run_jobs(
                eval_tasks, self._show_progress, self._workers
            )
            e2e_eval_dataset = QcapEvalDataset(samples=e2e_eval_samples)
            e2e_eval_dataset.save_json(self.evaluation_dataset_path)
            self.state_manager.mark_completed(DatasetState.E2E)
            logger.info("Evaluation dataset for end2end stage generated.")
