import yaml
import click
import logging
import time
import json
from pai_rag.evaluation.run_evaluation_pipeline import run_evaluation_pipeline


def run_experiment(exp_params):
    name = exp_params["name"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logging.info(
        f"Running experiment with name={name}, exp_params={exp_params}, timestamp={timestamp}"
    )
    try:
        # 运行实验并获取结果
        result = run_evaluation_pipeline(
            config=exp_params["setting_file"],
            data_path=exp_params["data_path"],
            name=name,
        )
        logging.info(f"Finished experiment with name={name}")

        # 记录实验结果到 JSON 文件
        result_filename = f"localdata/eval_exp_data/results_{name}_{timestamp}.json"
        with open(result_filename, "w") as result_file:
            json.dump(
                {"name": name, "parameters": exp_params, "result": result}, result_file
            )
        logging.info(f"Results saved to {result_filename}")

    except Exception as e:
        logging.error(f"Error running experiment {name}: {e}")


@click.command()
@click.option("-e", "--exp_config", show_default=True)
def run(exp_config=None):
    with open(exp_config) as file:
        configs = yaml.safe_load(file)

    for exp in configs["experiment"]:
        run_experiment(exp)
