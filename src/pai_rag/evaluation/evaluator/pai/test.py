import json
import pandas as pd
import asyncio
from pai.llm_eval.evals.models.openai_model import OpenAIModel
from pai.llm_eval.evals.default_templates import DefaultPromptTemplateCN
from pai.llm_eval.evals.evaluators import LLMEvaluator
from pai.llm_eval.evals.executors import run_evals

eval_model = OpenAIModel(model="gpt-3.5-turbo")

faithfulness_evaluator = LLMEvaluator(
    eval_model, DefaultPromptTemplateCN.FAITHFULNESS_PROMPT_TEMPLATE
)
correctness_evaluator = LLMEvaluator(
    eval_model, DefaultPromptTemplateCN.RAG_CORRECTNESS_PROMPT_TEMPLATE
)
evaluators = [faithfulness_evaluator, correctness_evaluator]


async def compute_response_metric(qca_dataset, evaluators):
    response_df = pd.DataFrame(qca_dataset)
    response_eval_df = response_df[
        ["query", "predicted_contexts", "predicted_answer"]
    ].rename(
        columns={
            "query": "input",
            "predicted_answer": "output",
            "predicted_contexts": "reference",
        }
    )
    faithfulness_eval_df, correctness_eval_df = run_evals(
        response_eval_df, evaluators, True, True, False, None, True
    )
    import pdb

    pdb.set_trace()
    return faithfulness_eval_df, correctness_eval_df


async def compute_response_metric_q(qca_dataset):
    response_df = pd.DataFrame(qca_dataset)
    response_eval_df = response_df[
        ["query", "predicted_contexts", "predicted_answer"]
    ].rename(
        columns={
            "query": "input",
            "predicted_answer": "output",
            "predicted_contexts": "reference",
        }
    )

    faithfulness_eval_df, correctness_eval_df = run_evals(
        response_eval_df, evaluators, True, True, False, None, True
    )
    import pdb

    pdb.set_trace()
    return faithfulness_eval_df, correctness_eval_df


async def aevaluation_for_response():
    qca_dataset_path = "/home/xiaowen/xiaowen/github_code/PAI-RAG/localdata/eval_exp_data/storage__exp1/predicted_qca_dataset.json"

    # Load your dataset
    with open(qca_dataset_path, "r", encoding="utf-8") as f:
        qcas = json.load(f)
        qcas = qcas["examples"][0:5]

    results = await compute_response_metric_q(qcas)
    print("Results:", results)


def run_evaluation_pipeline():
    asyncio.run(aevaluation_for_response())


def run_experiment():
    for i in range(2):
        run_evaluation_pipeline()


# run_experiment()
