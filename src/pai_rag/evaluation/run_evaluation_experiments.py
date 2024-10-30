import yaml
import click
import logging
import time
import json
import hashlib
from pai_rag.evaluation.run_evaluation_pipeline import run_evaluation_pipeline


def validate_json_file(ctx, param, value):
    """检查文件路径是否以 .json 结尾"""
    if value is not None and not value.endswith(".json"):
        raise click.BadParameter(
            "Output path must be a JSON file with a .json extension."
        )
    return value


def calculate_md5_from_json(data):
    """计算 JSON 内容的 MD5 值"""
    hasher = hashlib.md5()
    # 将 JSON 对象转换为字符串，并确保顺序一致
    json_str = json.dumps(data, sort_keys=True)
    hasher.update(json_str.encode("utf-8"))
    return hasher.hexdigest()


def run_experiment(exp_params):
    exp_name = exp_params["name"]
    logging.info(f"Running experiment with name={exp_name}, exp_params={exp_params}")
    try:
        # 运行实验并获取结果
        result = run_evaluation_pipeline(
            config=exp_params["rag_setting_file"],
            data_path=exp_params["eval_data_path"],
            exp_name=exp_name,
            eval_model_source=exp_params["eval_model_source"],
            eval_model_name=exp_params["eval_model_name"],
        )
        logging.info(f"Finished experiment with name={exp_name}")
    except Exception as e:
        logging.error(f"Error running experiment {exp_name}: {e}")

    return {"name": exp_params["name"], "parameters": exp_params, "result": result}


@click.command()
@click.option("-i", "--input_exp_config", show_default=True)
@click.option("-o", "--output_path", callback=validate_json_file, show_default=True)
def run(input_exp_config=None, output_path=None):
    with open(input_exp_config) as file:
        configs = yaml.safe_load(file)

    if not output_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_key = calculate_md5_from_json(configs)
        output_path = f"localdata/eval_exp_data/results_{file_key}_{timestamp}.json"
    results = []
    for exp in configs["experiment"]:
        result = run_experiment(exp)
        results.append(result)

    with open(output_path, "w") as result_file:
        json.dump(results, result_file, ensure_ascii=False, indent=4)

    logging.info(f"Results saved to {output_path}")
