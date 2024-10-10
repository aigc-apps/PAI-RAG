import yaml

import logging

# 初始化日志
logging.basicConfig(filename="experiment.log", level=logging.INFO)


def run_experiment(name, exp_params):
    logging.info(f"Running experiment with name={name}, exp_params={exp_params}")
    # 这里是实验代码
    # 例如模型训练，计算精度等
    accuracy = 0.90  # 示例
    logging.info(f"Finished with accuracy: {accuracy}")


# 读取配置文件
with open("config.yaml") as file:
    configs = yaml.safe_load(file)

# 运行所有实验
for exp in configs["experiment"]:
    run_experiment(exp["name"], exp)
