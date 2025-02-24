import os
import asyncio
from dotenv import load_dotenv
import pickle
import json
from loguru import logger
import sys

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pai_rag.integrations.llms.pai.llm_config import DashScopeLlmConfig
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.integrations.data_analysis.text2sql.evaluations.bird_evaluator import (
    BirdEvaluator,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 移除默认的日志处理器
logger.remove()
# 添加一个新的日志处理器，指定最低日志级别为 INFO，并输出到指定文件
log_file_path = "/tmp/log/bird/bird_eval.log"
logger.add(
    log_file_path, level="DEBUG", rotation="100 MB", retention="10 days", enqueue=True
)
# 添加一个可选的日志处理器，输出到控制台
logger.add(sys.stderr, level="INFO")

# 加载 .env 文件中的环境变量
load_dotenv()


if os.path.exists("./model_repository/bge-m3"):
    embed_model_bge = HuggingFaceEmbedding(
        model_name="./model_repository/bge-m3", embed_batch_size=20
    )
else:
    embed_model_bge = None

llm_config = DashScopeLlmConfig(max_tokens=2000, model="qwen-max")
llm = PaiLlm(llm_config)


if __name__ == "__main__":
    database_folder_path = "/tmp/datasets/BIRD/dev_20240627/dev_databases/"
    eval_file_path = "/tmp/datasets/BIRD/dev_20240627/dev.json"
    history_file_path = "/tmp/datasets/BIRD/train/train.json"
    analysis_config = {
        "enable_enhanced_description": False,
        "enable_db_history": True,
        "enable_db_embedding": True,
        "max_col_num": 100,
        "max_val_num": 1000,
        "enable_query_preprocessor": True,
        "enable_db_preretriever": True,
        "enable_db_selector": True,
    }

    bird_eval = BirdEvaluator(
        llm=llm,
        embed_model=embed_model_bge,
        database_folder_path=database_folder_path,
        eval_file_path=eval_file_path,
        analysis_config=analysis_config,
        history_file_path=history_file_path,
    )

    # batch_load
    asyncio.run(bird_eval.abatch_loader())

    # batch_predict
    predicted_sql_list, queried_result_list, db_id_list = asyncio.run(
        bird_eval.abatch_query(num_start=0, num_end=10)
    )

    # 写入二进制文件
    with open(
        "/tmp/text2sql_evaluation/bird/predicted_sql_list.pkl",
        "wb",
    ) as file:
        pickle.dump(predicted_sql_list, file)

    # save result
    predicted_file = "/tmp/text2sql_evaluation/bird/predict_dev.json"
    directory = os.path.dirname(predicted_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(predicted_file, "w", encoding="utf-8") as file:
        content = {}
        for idx, item in enumerate(predicted_sql_list):
            if item:
                parsed_item = bird_eval.parse_predicted_sql(item)
            content.update({idx: f"{parsed_item}\t----- bird -----\t{db_id_list[idx]}"})
        json.dump(content, file, ensure_ascii=False, indent=4)
    print(f"predicted_sql_list have been written to {predicted_file}")

    # batch_evaluate
    gold_file = "/tmp/datasets/BIRD/dev_20240627/"
    bird_eval.batch_evaluate(
        predicted_sql_path=predicted_file,
        ground_truth_path=gold_file,
        db_root_path=database_folder_path,
        diff_json_path=eval_file_path,
    )
