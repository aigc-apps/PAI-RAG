import yaml
import click
import logging
import time
import json
import hashlib
from pai_rag.evaluation.run_evaluation_pipeline import run_evaluation_pipeline


def calculate_md5_from_json(data):
    """计算 JSON 内容的 MD5 值"""
    hasher = hashlib.md5()
    # 将 JSON 对象转换为字符串，并确保顺序一致
    json_str = json.dumps(data, sort_keys=True)
    hasher.update(json_str.encode("utf-8"))
    return hasher.hexdigest()


def run_experiment(exp_params):
    name = exp_params["name"]
    logging.info(f"Running experiment with name={name}, exp_params={exp_params}")
    try:
        # 运行实验并获取结果
        result = run_evaluation_pipeline(
            config=exp_params["setting_file"],
            data_path=exp_params["data_path"],
            name=name,
        )
        logging.info(f"Finished experiment with name={name}")
    except Exception as e:
        logging.error(f"Error running experiment {name}: {e}")

    return {"name": exp_params["name"], "parameters": exp_params, "result": result}


@click.command()
@click.option("-e", "--exp_config", show_default=True)
def run(exp_config=None):
    with open(exp_config) as file:
        configs = yaml.safe_load(file)

    # 记录实验结果到 JSON 文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_key = calculate_md5_from_json(configs)
    result_filename = f"localdata/eval_exp_data/results_{file_key}_{timestamp}.json"
    results = []
    for exp in configs["experiment"]:
        result = run_experiment(exp)
        results.append(result)

    with open(result_filename, "w") as result_file:
        json.dump(results, result_file, indent=4)

    logging.info(f"Results saved to {result_filename}")
