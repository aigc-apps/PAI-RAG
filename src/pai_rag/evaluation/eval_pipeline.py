import asyncio
import click
import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry

_BASE_DIR = Path(__file__).parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")


class EvalDatasetPipeline:
    def __init__(self, evaluation):
        self.evaluation = evaluation

    async def eval_open_dataset(self, overwrite: bool, type: str, dataset_name: None):
        _ = await self.evaluation.aload_question_answer_pairs_json(
            overwrite, dataset_name
        )
        df, eval_res_avg = await self.evaluation.abatch_retrieval_response_aevaluation(
            type=type, workers=4, overwrite=overwrite
        )
        print("eval_res_avg", eval_res_avg)
        return df, eval_res_avg


def __init_eval_pipeline(config_file):
    config = RagConfiguration.from_file(config_file).get_value()
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
    asyncio.run(
        eval_pipeline.eval_open_dataset(
            overwrite=overwrite, type=type, dataset_name=name
        )
    )
