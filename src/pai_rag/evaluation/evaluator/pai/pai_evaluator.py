import os
import json
from pai_rag.evaluation.evaluator.retrieval_metrics import HitRate, MRR
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
    load_qca_dataset_json,
)
from pai.llm_eval.evals.evaluators import LLMEvaluator
from pai.llm_eval.evals.default_templates import DefaultPromptTemplateCN
from pai_rag.evaluation.evaluator.pai.test import compute_response_metric


class PaiEvaluator:
    def __init__(self, llm, persist_path: str = None):
        self._llm = llm
        self.persist_path = persist_path
        self.hitrate = HitRate()
        self.mrr = MRR()
        self.retrieval_evaluators = [self.hitrate, self.mrr]
        self.faithfulness_evaluator = LLMEvaluator(
            self._llm, DefaultPromptTemplateCN.FAITHFULNESS_PROMPT_TEMPLATE
        )
        self.correctness_evaluator = LLMEvaluator(
            self._llm, DefaultPromptTemplateCN.RAG_CORRECTNESS_PROMPT_TEMPLATE
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

    # async def compute_response_metric(self,qca_dataset):
    #     response_df = pd.DataFrame(qca_dataset)
    #     response_eval_df = response_df[['query', 'predicted_contexts', 'predicted_answer']].rename(columns={'query': 'input', 'predicted_answer': 'output', 'predicted_contexts': 'reference'})

    #     faithfulness_eval_df, correctness_eval_df = run_evals(
    #         response_eval_df,
    #         self.response_evaluators,
    #         True,
    #         True,
    #         False,
    #         None,
    #         True)
    #     import pdb
    #     pdb.set_trace()
    #     return faithfulness_eval_df, correctness_eval_df

    async def aevaluation_for_retrieval(self, qca_dataset):
        """Run retrieval evaluation with qca dataset."""
        retrieval_evaluation_dataset_path = os.path.join(
            self.persist_path, "retrieval_evaluation_dataset.json"
        )
        if os.path.exists(retrieval_evaluation_dataset_path):
            print(
                f"A evaluation dataset for retrieval already exists at {retrieval_evaluation_dataset_path}."
            )
            retrieval_evaluation_dataset = load_qca_dataset_json(
                retrieval_evaluation_dataset_path
            )
            examples = retrieval_evaluation_dataset["examples"]
            mean_result = {
                "hitrate": sum(float(entry["hitrate"]) for entry in examples)
                / len(examples),
                "mrr": sum(float(entry["mrr"]) for entry in examples) / len(examples),
            }
            return mean_result
        else:
            print("Starting to generate evaluation dataset for retrieval...")
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
                {"hitrate": h, "mrr": m}
                for h, m in zip(result["hitrate"], result["mrr"])
            ]
            for (
                qca,
                answer_response,
            ) in zip(qca_dataset, responses):
                combined_dict = {**qca, **answer_response}
                example = RetrievalEvaluationSample(**combined_dict)
                examples.append(example)
            retrieval_evaluation_dataset = LabelledRagDataset(examples=examples)
            save_qca_dataset_json(
                retrieval_evaluation_dataset, retrieval_evaluation_dataset_path
            )
            mean_result = {
                key: sum(values) / len(values) for key, values in result.items()
            }
            return mean_result

    async def aevaluation_for_response(self, qca_dataset):
        """Run response evaluation with qca dataset."""
        response_evaluation_dataset_path = os.path.join(
            self.persist_path, "response_evaluation_dataset.json"
        )
        if os.path.exists(response_evaluation_dataset_path):
            print(
                f"A evaluation dataset for response already exists at {response_evaluation_dataset_path}."
            )
            response_evaluation_dataset = load_qca_dataset_json(
                response_evaluation_dataset_path
            )
            examples = response_evaluation_dataset["examples"]
            mean_result = {
                "faithfulness_score": sum(
                    float(entry["faithfulness_score"]) for entry in examples
                )
                / len(examples),
                "correctness_score": sum(
                    float(entry["correctness_score"]) for entry in examples
                )
                / len(examples),
            }
            return mean_result
        else:
            print("Starting to generate evaluation dataset for response...")
            examples = []
            result = {}
            faithfulness_eval_df, correctness_eval_df = await compute_response_metric(
                qca_dataset, self.response_evaluators
            )
            responses = [
                {
                    "faithfulness_score": f_s,
                    "faithfulness_reason": f_e,
                    "correctness_score": c_s,
                    "correctness_reason": c_e,
                }
                for f_s, f_e, c_s, c_e in zip(
                    faithfulness_eval_df["score"].tolist(),
                    faithfulness_eval_df["explanation"].tolist(),
                    correctness_eval_df["score"].tolist(),
                    correctness_eval_df["explanation"].tolist(),
                )
            ]
            for (
                qca,
                answer_response,
            ) in zip(qca_dataset, responses):
                combined_dict = {**qca, **answer_response}
                example = ResponseEvaluationSample(**combined_dict)
                examples.append(example)

            response_evaluation_dataset = LabelledRagDataset(examples=examples)
            save_qca_dataset_json(
                response_evaluation_dataset, response_evaluation_dataset_path
            )
            mean_result = {
                key: sum(value[0] for value in values) / len(values)
                for key, values in result.items()
            }
            return mean_result
