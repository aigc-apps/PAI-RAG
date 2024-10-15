import os
import json
from pai_rag.evaluation.evaluator.retrieval_metrics import HitRate, MRR
from pai_rag.evaluation.evaluator.response_faithfulness_metric import (
    FaithfulnessEvaluator,
)
from pai_rag.evaluation.evaluator.response_correctness_metric import (
    CorrectnessEvaluator,
)
from pai_rag.evaluation.generator.rag_qca_sample import (
    RetrievalEvaluationSample,
    ResponseEvaluationSample,
)
from llama_index.core.async_utils import run_jobs
from llama_index.core.llama_dataset import (
    LabelledRagDataset,
)
from pai_rag.evaluation.generator.utils import (
    save_qca_dataset_json,
)


class BaseEvaluator:
    def __init__(self, llm, persist_path: str = None):
        self._llm = llm
        self.persist_path = persist_path
        self.hitrate = HitRate()
        self.mrr = MRR()
        self.retrieval_evaluators = [self.hitrate, self.mrr]
        self.faithfulness_evaluator = FaithfulnessEvaluator(
            llm=self._llm,
        )
        self.correctness_evaluator = CorrectnessEvaluator(
            llm=self._llm,
        )
        self.response_evaluators = [
            self.faithfulness_evaluator,
            self.correctness_evaluator,
        ]
        self._show_progress = True
        self._workers = 2

    def load_predicted_qca_dataset(self) -> None:
        labelled_qca_dataset_path = os.path.join(
            self.persist_path, "predicted_qca_dataset.json"
        )
        with open(labelled_qca_dataset_path, "r", encoding="utf-8") as f:
            qcas = json.load(f)
            return qcas["examples"][0:5]

    async def compute_retrieval_metric(
        self, metric, labelled_node_ids, predicted_node_ids
    ):
        eval_result = metric.compute(labelled_node_ids, predicted_node_ids)
        return eval_result

    async def compute_response_metric(self, metric, query, response, contexts):
        eval_result = await metric.aevaluate(query, response, contexts)
        return eval_result

    async def aevaluation_for_retrieval(self, qca_dataset):
        """Run retrieval evaluation with qca dataset."""
        result = {}
        examples = []
        for metric in self.retrieval_evaluators:
            eval_tasks = []
            for qca in qca_dataset:
                eval_tasks.append(
                    self.compute_retrieval_metric(
                        metric, qca["reference_node_id"], qca["predicted_node_id"]
                    )
                )
            metric_result = await run_jobs(
                eval_tasks, self._show_progress, self._workers
            )
            result[metric.metric_name] = metric_result
        responses = [
            {"hitrate": h, "mrr": m} for h, m in zip(result["hitrate"], result["mrr"])
        ]
        for (
            qca,
            answer_response,
        ) in zip(qca_dataset, responses):
            combined_dict = {**qca, **answer_response}
            example = RetrievalEvaluationSample(**combined_dict)
            examples.append(example)
        retrieval_evaluation_dataset = LabelledRagDataset(examples=examples)
        retrieval_evaluation_dataset_path = os.path.join(
            self.persist_path, "retrieval_evaluation_dataset.json"
        )
        save_qca_dataset_json(
            retrieval_evaluation_dataset, retrieval_evaluation_dataset_path
        )
        mean_result = {key: sum(values) / len(values) for key, values in result.items()}
        return mean_result

    async def aevaluation_for_response(self, qca_dataset):
        """Run response evaluation with qca dataset."""
        examples = []
        result = {}
        for metric in self.response_evaluators:
            eval_tasks = []
            for qca in qca_dataset:
                eval_tasks.append(
                    self.compute_response_metric(
                        metric,
                        qca["query"],
                        qca["reference_answer"],
                        qca["predicted_contexts"],
                    )
                )
            metric_result = await run_jobs(
                eval_tasks, self._show_progress, self._workers
            )
            result[metric.metric_name] = metric_result
        responses = [
            {
                "faithfulness_score": f[0],
                "faithfulness_reason": f[1],
                "correctness_score": c[0],
                "correctness_reason": c[1],
            }
            for f, c in zip(result["faithfulness"], result["correctness"])
        ]
        for (
            qca,
            answer_response,
        ) in zip(qca_dataset, responses):
            combined_dict = {**qca, **answer_response}
            example = ResponseEvaluationSample(**combined_dict)
            examples.append(example)
        response_evaluation_dataset = LabelledRagDataset(examples=examples)
        response_evaluation_dataset_path = os.path.join(
            self.persist_path, "response_evaluation_dataset.json"
        )
        save_qca_dataset_json(
            response_evaluation_dataset, response_evaluation_dataset_path
        )
        mean_result = {
            key: sum(value[0] for value in values) / len(values)
            for key, values in result.items()
        }
        return mean_result
