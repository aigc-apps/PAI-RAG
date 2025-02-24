import os
import asyncio
from dotenv import load_dotenv
import pickle
import json
from loguru import logger
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pai_rag.integrations.llms.pai.llm_config import DashScopeLlmConfig
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.integrations.data_analysis.text2sql.evaluations.bird_evaluator import (
    BirdEvaluator,
)

# 移除默认的日志处理器
logger.remove()
# 添加一个新的日志处理器，指定最低日志级别为 INFO，并输出到指定文件
log_file_path = "/tmp/log/bird/bird_eval_para.log"
logger.add(
    log_file_path, level="DEBUG", rotation="100 MB", retention="10 days", enqueue=True
)
# 添加一个可选的日志处理器，输出到控制台
logger.add(sys.stderr, level="INFO")

# 加载 .env 文件中的环境变量
load_dotenv()

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


def setup_bird_eval(
    embed_model,
    llm,
    database_folder_path,
    eval_file_path,
    history_file_path,
    analysis_config,
):
    bird_eval = BirdEvaluator(
        llm=llm,
        embed_model=embed_model,
        database_folder_path=database_folder_path,
        eval_file_path=eval_file_path,
        analysis_config=analysis_config,
        history_file_path=history_file_path,
    )
    return bird_eval


def initialize_resources():
    if os.path.exists("./model_repository/bge-m3"):
        embed_model_bge = HuggingFaceEmbedding(
            model_name="./model_repository/bge-m3", embed_batch_size=20
        )
    else:
        embed_model_bge = None

    llm_config = DashScopeLlmConfig(max_tokens=2000, model="qwen-max")
    llm = PaiLlm(llm_config)
    # llm.max_retries = 5
    # llm.timeout = 360.0
    return embed_model_bge, llm


def abatch_load_wrapper(
    database_folder_path, eval_file_path, history_file_path, analysis_config
):
    embed_model_bge, llm = initialize_resources()
    bird_eval = setup_bird_eval(
        embed_model_bge,
        llm,
        database_folder_path,
        eval_file_path,
        history_file_path,
        analysis_config,
    )
    return asyncio.run(bird_eval.abatch_loader())


def abatch_query_wrapper(
    start_idx,
    num_queries,
    database_folder_path,
    eval_file_path,
    history_file_path,
    analysis_config,
):
    embed_model_bge, llm = initialize_resources()
    bird_eval = setup_bird_eval(
        embed_model_bge,
        llm,
        database_folder_path,
        eval_file_path,
        history_file_path,
        analysis_config,
    )
    try:
        predicted_sql_list, queried_result_list, db_id_list = asyncio.run(
            bird_eval.abatch_query(num_start=start_idx, num_end=start_idx + num_queries)
        )
        logger.info(
            f"Completed queries from index {start_idx} to {start_idx + num_queries}"
        )
    except asyncio.TimeoutError:
        logger.error(
            f"Timeout occurred while processing queries from index {start_idx} to {start_idx + num_queries}"
        )
        return []
    except Exception as e:
        logger.error(
            f"Exception occurred while processing queries from index {start_idx} to {start_idx + num_queries}: {e}"
        )
        return []
    # 返回带索引的结果
    return list(
        zip(
            range(start_idx, start_idx + len(predicted_sql_list)),
            predicted_sql_list,
            queried_result_list,
            db_id_list,
        )
    )


if __name__ == "__main__":
    # 设置进程数
    num_processes = 4
    total_queries = 1534
    queries_per_process = total_queries // num_processes

    embed_model_bge, llm = initialize_resources()
    bird_eval = setup_bird_eval(
        embed_model_bge,
        llm,
        database_folder_path,
        eval_file_path,
        history_file_path,
        analysis_config,
    )
    # asyncio.run(bird_eval.abatch_loader())

    # 使用 ProcessPoolExecutor 并行执行 abatch_load
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                abatch_load_wrapper,
                database_folder_path,
                eval_file_path,
                history_file_path,
                analysis_config,
            )
            for _ in range(num_processes)
        ]

    # 使用 ProcessPoolExecutor 并行执行 abatch_query
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                abatch_query_wrapper,
                i * queries_per_process,
                queries_per_process,
                database_folder_path,
                eval_file_path,
                history_file_path,
                analysis_config,
            )
            for i in range(num_processes)
        ]

        results = [future.result() for future in futures]

    # 合并结果并按索引排序
    combined_results = []
    for future in as_completed(futures, timeout=600):  # 设置超时时间
        try:
            result = future.result(timeout=300)  # 设置超时时间
            combined_results.extend(result)
        except TimeoutError:
            logger.error("Timeout occurred while waiting for a future to complete.")
        except Exception as e:
            logger.error(f"Exception occurred: {e}")

    combined_results.sort(key=lambda x: x[0])  # 按索引排序

    # 提取排序后的预测SQL、查询结果和数据库ID
    sorted_predicted_sql_list = [item[1] for item in combined_results]
    sorted_queried_result_list = [item[2] for item in combined_results]
    sorted_db_id_list = [item[3] for item in combined_results]

    # 写入二进制文件
    with open(
        "/tmp/text2sql_evaluation/bird/predicted_sql_list.pkl",
        "wb",
    ) as file:
        pickle.dump(sorted_predicted_sql_list, file)

    # 保存结果
    predicted_file = "/tmp/text2sql_evaluation/bird/predict_dev.json"
    directory = os.path.dirname(predicted_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(predicted_file, "w", encoding="utf-8") as file:
        content = {}
        for idx, item in enumerate(sorted_predicted_sql_list):
            if item:
                parsed_item = bird_eval.parse_predicted_sql(item)
            content.update(
                {idx: f"{parsed_item}\t----- bird -----\t{sorted_db_id_list[idx]}"}
            )
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
