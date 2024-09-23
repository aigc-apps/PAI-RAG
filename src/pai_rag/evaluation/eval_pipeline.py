import asyncio
import click
import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from dynaconf import Dynaconf

_BASE_DIR = Path(__file__).parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")


class EvalDatasetPipeline:
    def __init__(self, evaluation):
        self.evaluation = evaluation

    def load_eval_dataset(self):
        self.evaluation.load_evaluation_dataset()

    async def generate_qa_from_custom_dataset(self):
        qa_dataset = await self.evaluation.aload_question_answer_pairs_json()
        return qa_dataset

    async def generate_qa_from_open_dataset(self, dataset_name: None):
        qa_dataset = await self.evaluation.aload_question_answer_pairs_json(
            dataset_name=dataset_name
        )
        return qa_dataset

    async def eval_qa_from_custom_dataset(
        self, overwrite: bool, type: str, dataset_name: str = None
    ):
        qa_dataset = await self.evaluation.aload_question_answer_pairs_json(
            overwrite, dataset_name
        )
        print("qa_dataset", qa_dataset)
        df, eval_res_avg = await self.evaluation.abatch_retrieval_response_aevaluation(
            type=type, workers=4, overwrite=overwrite
        )
        print("df", df)
        print("eval_res_avg", eval_res_avg)
        return df, eval_res_avg

    async def eval_open_dataset(self, overwrite: bool, type: str, dataset_name: None):
        _ = await self.evaluation.aload_question_answer_pairs_json(
            overwrite, dataset_name
        )
        df, eval_res_avg = await self.evaluation.abatch_retrieval_response_aevaluation(
            type=type, workers=4, overwrite=overwrite
        )
        return df, eval_res_avg


def __init_eval_pipeline(config_file):
    config = RagConfiguration.from_file(config_file).get_value()
    snapshot_config = Dynaconf(settings_file=[config_file])
    config.update(snapshot_config, tomlfy=True, merge=True)

    module_registry.init_modules(config)

    evaluation = module_registry.get_module_with_config("EvaluationModule", config)

    return EvalDatasetPipeline(evaluation)


@click.command()
@click.option(
    "-c",
    "--config",
    show_default=True,
    help=f"Configuration file. Default: {DEFAULT_APPLICATION_CONFIG_FILE}",
    default=DEFAULT_APPLICATION_CONFIG_FILE,
)
@click.option(
    "-o",
    "--overwrite",
    show_default=True,
    help="Whether to overwrite the generated QA file",
    default=False,
)
@click.option(
    "-t",
    "--type",
    show_default=True,
    help="Evaluation types in [all, retrieval, response]",
    default="all",
)
@click.option(
    "-n",
    "--name",
    show_default=True,
    help="Open Dataset Name. Optional: [miracl, duretrieval]",
    default=None,
)
def run(config, overwrite, type, name):
    eval_pipeline = __init_eval_pipeline(config)
    # asyncio.run(
    #     eval_pipeline.eval_open_dataset(
    #         overwrite=overwrite, type=type, dataset_name=name
    #     )
    # )
    eval_pipeline.load_eval_dataset()
    asyncio.run(
        eval_pipeline.eval_qa_from_custom_dataset(overwrite=False, type="retrieval")
    )
