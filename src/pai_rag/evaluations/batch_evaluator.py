"""Evaluator for both Retrieval & Response based on config"""

import click
import logging
from typing import Optional
import pandas as pd
import asyncio
import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    FaithfulnessEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
)
from pai_rag.evaluations.dataset_generation.generate_question_answer_pairs import (
    load_question_answer_pairs_json,
)
from pai_rag.evaluations.batch_eval_runner import BatchEvalRunner

DEFAULT_QA_DATASET_DIR = "localdata/evaluation"
logger = logging.getLogger(__name__)


class BatchEvaluator(object):
    def __init__(
        self,
        config,
        retriever,
        query_engine,
        llm=None,
    ):
        self.name = "BatchEvaluator"
        self.config = config
        self.llm = llm
        self.dataset_path = config.get(
            "qa_dataset_path", os.path.join(DEFAULT_QA_DATASET_DIR, "qa_dataset.json")
        )
        self.query_engine = query_engine
        # retrieval
        self.retrieval_metrics = config["evaluation"]["retrieval"]
        self.retrieval_evaluator = RetrieverEvaluator.from_metric_names(
            self.retrieval_metrics, retriever=retriever
        )
        # response
        self.response_metrics_list = self.config["evaluation"]["response"]
        self.faithfulness = FaithfulnessEvaluator()
        self.answer_relevancy = AnswerRelevancyEvaluator()
        self.correctness = CorrectnessEvaluator()
        self.similarity = SemanticSimilarityEvaluator()

        logger.info("batch evaluator created")

    async def batch_retrieval_response_aevaluation(
        self,
        type: Optional[str] = "all",
        workers: Optional[int] = 2,
        save_to_file: Optional[bool] = True,
        overwrite: Optional[bool] = False,
        output: Optional[str] = None,
    ):
        # generate or load qa_dataset
        qas = await load_question_answer_pairs_json(self.dataset_path, overwrite)
        data = {
            "query": [t["query"] for t in qas["examples"]],
            "reference_contexts": [t["reference_contexts"] for t in qas["examples"]],
            "reference_node_id": [t["reference_node_id"] for t in qas["examples"]],
            "reference_answer": [t["reference_answer"] for t in qas["examples"]],
        }

        df = pd.DataFrame(data)
        # retrieval and response evaluation based on the same dataset"""
        queries = df["query"].tolist()
        reference_node_ids = df["reference_node_id"].tolist()
        reference_answers = df["reference_answer"].tolist()
        retrieval_evaluator = {}
        if type in ["retrieval", "all"]:
            retrieval_evaluator["retrieval"] = self.retrieval_evaluator

        response_evaluators = {}
        if type in ["response", "all"]:
            if "Faithfulness" in self.response_metrics_list:
                response_evaluators["Faithfulness"] = self.faithfulness
            if "Answer Relevancy" in self.response_metrics_list:
                response_evaluators["Answer Relevancy"] = self.answer_relevancy
            if "Correctness" in self.response_metrics_list:
                response_evaluators["Correctness"] = self.correctness
            if "Semantic Similarity" in self.response_metrics_list:
                response_evaluators["Semantic Similarity"] = self.similarity

        runner = BatchEvalRunner(
            retrieval_evaluator,
            response_evaluators,
            workers=workers,
            show_progress=True,
        )
        eval_results = await runner.aevaluate_queries(
            query_engine=self.query_engine,
            queries=queries,
            node_ids=reference_node_ids,
            reference_answers=reference_answers,
        )

        if type in ["retrieval", "all"]:
            df["retrieval_contexts"] = [
                e.retrieved_texts for e in eval_results["retrieval"]
            ]
            df["retrieval_node_ids"] = [
                e.retrieved_ids for e in eval_results["retrieval"]
            ]
            if "hit_rate" in self.retrieval_metrics:
                df["hit_rate"] = [
                    e.metric_vals_dict["hit_rate"] for e in eval_results["retrieval"]
                ]
            else:
                df["hit_rate"] = "not selected"
            if "mrr" in self.retrieval_metrics:
                df["mrr"] = [
                    e.metric_vals_dict["mrr"] for e in eval_results["retrieval"]
                ]
            else:
                df["mrr"] = "not selected"
        if type in ["response", "all"]:
            if "Faithfulness" in self.response_metrics_list:
                df["faithfulness_score"] = [
                    e.score for e in eval_results["Faithfulness"]
                ]
                df["response_answer"] = [
                    e.response for e in eval_results["Faithfulness"]
                ]
            else:
                df["faithfulness_score"] = "not selected"
            if "Answer Relevancy" in self.response_metrics_list:
                df["answer_relevancy_score"] = [
                    e.feedback for e in eval_results["Answer Relevancy"]
                ]
            else:
                df["answer_relevancy_score"] = "not selected"
            if "Correctness" in self.response_metrics_list:
                df["correctness_score"] = [e.score for e in eval_results["Correctness"]]
            else:
                df["correctness_score"] = "not selected"
            if "Semantic Similarity" in self.response_metrics_list:
                df["semantic_similarity_score"] = [
                    e.score for e in eval_results["Semantic Similarity"]
                ]
            else:
                df["semantic_similarity_score"] = "not selected"

        if type == "retrieval":
            eval_res_avg = {
                "batch_number": df.shape[0],
                "hit_rate_mean": df["hit_rate"].agg("mean"),
                "mrr_mean": df["mrr"].agg("mean"),
            }
        elif type == "response":
            eval_res_avg = {
                "batch_number": df.shape[0],
                "faithfulness_mean": df["faithfulness_score"].agg("mean"),
                "correctness_mean": df["correctness_score"].agg("mean"),
                "similarity_mean": df["semantic_similarity_score"].agg("mean"),
            }
        else:
            eval_res_avg = {
                "batch_number": df.shape[0],
                "hit_rate_mean": df["hit_rate"].agg("mean"),
                "mrr_mean": df["mrr"].agg("mean"),
                "faithfulness_mean": df["faithfulness_score"].agg("mean"),
                "correctness_mean": df["correctness_score"].agg("mean"),
                "similarity_mean": df["semantic_similarity_score"].agg("mean"),
            }

        if save_to_file:
            if output is None or output == "":
                if not os.path.exists(DEFAULT_QA_DATASET_DIR):
                    os.makedirs(DEFAULT_QA_DATASET_DIR, exist_ok=True)
                output = os.path.join(DEFAULT_QA_DATASET_DIR, "batch_eval_results.xlsx")
            df.to_excel(output)
        return df, eval_res_avg


def __init_evaluator_pipeline():
    base_dir = Path(__file__).parent.parent
    config_file = os.path.join(base_dir, "config/settings.yaml")

    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)

    retriever = module_registry.get_module_with_config("RetrieverModule", config)
    query_engine = module_registry.get_module_with_config("QueryEngineModule", config)

    return BatchEvaluator(config, retriever, query_engine)


async def async_run(type, overwrite, output):
    print(f"Running async task with evaluation type: {type}")
    evaluation = __init_evaluator_pipeline()
    df, eval_res_avg = await evaluation.batch_retrieval_response_aevaluation(
        type=type, workers=2, save_to_file=True, overwrite=overwrite, output=output
    )
    return df, eval_res_avg


@click.command()
@click.option(
    "-t",
    "--type",
    show_default=True,
    help="Evaluation type: [retrieval, response, all]",
    default="all",
)
@click.option(
    "-o",
    "--overwrite",
    show_default=True,
    help="Whether to regenerate and overwrite the qa_dataset file: [True, False]",
    default=False,
)
@click.option(
    "-f",
    "--file_path",
    show_default=True,
    help="The output path of the generated evaluation result file",
    default=None,
)
def run(type, overwrite, file_path):
    df, eval_res_avg = asyncio.run(async_run(type, overwrite, file_path))
    print("Evaluation results is:", eval_res_avg)
