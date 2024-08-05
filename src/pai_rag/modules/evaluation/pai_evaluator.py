import logging
import os
from typing import Optional
import json
import pandas as pd

# from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    FaithfulnessEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
)
from pai_rag.modules.evaluation.ragdataset_generator import GenerateDatasetPipeline
from pai_rag.modules.evaluation.batch_eval_runner import BatchEvalRunner
from pai_rag.integrations.evaluation.retrieval.evaluator import MyRetrieverEvaluator

logger = logging.getLogger(__name__)
DEFAULT_QA_DATASET_DIR = "localdata/evaluation"


class PaiEvaluator:
    def __init__(
        self,
        llm,
        index,
        query_engine,
        retriever,
        retrieval_metrics,
        response_metrics,
    ):
        self.llm = llm
        self.index = index
        self.query_engine = query_engine
        self.retriever = retriever
        self.retrieval_metrics = retrieval_metrics
        self.retrieval_evaluator = MyRetrieverEvaluator.from_metric_names(
            self.retrieval_metrics, retriever=self.retriever
        )
        # TODO: if exists in list
        self.response_metrics = response_metrics
        self.faithfulness = FaithfulnessEvaluator()
        self.answer_relevancy = AnswerRelevancyEvaluator()
        self.correctness = CorrectnessEvaluator()
        self.similarity = SemanticSimilarityEvaluator()

        if not os.path.exists(DEFAULT_QA_DATASET_DIR):
            os.makedirs(DEFAULT_QA_DATASET_DIR, exist_ok=True)
        self.dataset_path = os.path.join(DEFAULT_QA_DATASET_DIR, "qa_dataset.json")

        logger.info("PaiEvaluator initialized.")

    async def aload_question_answer_pairs_json(
        self, overwrite: bool = False, dataset_name=None
    ):
        file_exists = os.path.exists(self.dataset_path)
        if file_exists and not overwrite:
            logging.info(
                f"[Evaluation] qa_dataset '{self.dataset_path}' already exists, do not need to regenerate and overwrite."
            )
        else:
            logging.info(
                f"[Evaluation] qa_dataset '{self.dataset_path}' (re)generating and overwriting..."
            )
            directory = os.path.dirname(self.dataset_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            await self.acustomized_generate_qas(dataset_name)

        logging.info(
            f"[Evaluation] loading generated qa_dataset from path {self.dataset_path}. "
        )
        with open(self.dataset_path) as f:
            qa_dataset = json.load(f)
        return qa_dataset

    async def acustomized_generate_qas(self, dataset_name=None):
        docs = self.index.vector_index._docstore.docs
        nodes = list(docs.values())
        pipeline = GenerateDatasetPipeline(llm=self.llm, nodes=nodes)
        if not dataset_name:
            qas = await pipeline.agenerate_dataset()
        else:
            print(f"Generating ragdataset for open dataset_name {dataset_name}")
            qas = await pipeline.generate_dataset_from_opendataset(dataset_name)
        pipeline.save_json(qas, self.dataset_path)

    async def abatch_retrieval_response_aevaluation(
        self,
        type: Optional[str] = "all",
        workers: Optional[int] = 2,
        output_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        # generate or load qa_dataset
        qas = await self.aload_question_answer_pairs_json(overwrite)
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
            if "Faithfulness" in self.response_metrics:
                response_evaluators["Faithfulness"] = self.faithfulness
            if "Answer Relevancy" in self.response_metrics:
                response_evaluators["Answer Relevancy"] = self.answer_relevancy
            if "Correctness" in self.response_metrics:
                response_evaluators["Correctness"] = self.correctness
            if "Semantic Similarity" in self.response_metrics:
                response_evaluators["Semantic Similarity"] = self.similarity

        runner = BatchEvalRunner(
            retrieval_evaluator,
            response_evaluators,
            workers=workers,
            show_progress=True,
        )
        if type in ["response", "all"]:
            eval_results = await runner.aevaluate_queries(
                query_engine=self.query_engine,
                queries=queries,
                node_ids=reference_node_ids,
                reference_answers=reference_answers,
            )
        else:
            eval_results = await runner.aevaluate_queries_for_retrieval(
                retriever=self.retriever,
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
            df["retrieval_node_scores"] = [
                e.retrieved_scores for e in eval_results["retrieval"]
            ]
            if "hit_rate" in self.retrieval_metrics:
                df["hit_rate"] = [
                    e.metric_vals_dict["hit_rate"] for e in eval_results["retrieval"]
                ]
            else:
                df["hit_rate"] = None
            if "mrr" in self.retrieval_metrics:
                df["mrr"] = [
                    e.metric_vals_dict["mrr"] for e in eval_results["retrieval"]
                ]
            else:
                df["mrr"] = None
        if type in ["response", "all"]:
            df["response_answer"] = None
            if "Faithfulness" in self.response_metrics:
                df["faithfulness_score"] = [
                    e.score for e in eval_results["Faithfulness"]
                ]
                df["response_answer"] = [
                    e.response for e in eval_results["Faithfulness"]
                ]
            else:
                df["faithfulness_score"] = None
            if "Answer Relevancy" in self.response_metrics:
                df["answer_relevancy_score"] = [
                    e.feedback for e in eval_results["Answer Relevancy"]
                ]
                if df["response_answer"][0] is None:
                    df["response_answer"] = [
                        e.response for e in eval_results["Answer Relevancy"]
                    ]
            else:
                df["answer_relevancy_score"] = None
            if "Correctness" in self.response_metrics:
                df["correctness_score"] = [e.score for e in eval_results["Correctness"]]
                if df["response_answer"][0] is None:
                    df["response_answer"] = [
                        e.response for e in eval_results["Correctness"]
                    ]
            else:
                df["correctness_score"] = None
            if "Semantic Similarity" in self.response_metrics:
                df["semantic_similarity_score"] = [
                    e.score for e in eval_results["Semantic Similarity"]
                ]
                if df["response_answer"][0] is None:
                    df["response_answer"] = [
                        e.response for e in eval_results["Correctness"]
                    ]
            else:
                df["semantic_similarity_score"] = None

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

        if output_path is None or output_path == "":
            output_path = os.path.join(
                DEFAULT_QA_DATASET_DIR, f"batch_eval_results_{type}.xlsx"
            )
        df.to_excel(output_path)
        return df, eval_res_avg
