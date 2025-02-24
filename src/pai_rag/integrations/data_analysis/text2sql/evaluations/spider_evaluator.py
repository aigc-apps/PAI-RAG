from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import json
import asyncio
from loguru import logger

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import QueryBundle

from pai_rag.integrations.data_analysis.data_analysis_config import SqliteAnalysisConfig
from pai_rag.integrations.data_analysis.text2sql.db_connector import SqliteConnector

from pai_rag.integrations.data_analysis.text2sql.db_loader import DBLoader
from pai_rag.integrations.data_analysis.text2sql.db_query import DBQuery

from pai_rag.integrations.data_analysis.text2sql.evaluations.eval_spider.evaluation import (
    build_foreign_key_map_from_json,
    evaluate,
)
from pai_rag.integrations.data_analysis.data_analysis_tool import (
    resolve_history_retriever,
    resolve_schema_retriever,
    resolve_value_retriever,
)
from pai_rag.integrations.data_analysis.text2sql.evaluations.base_evaluator import (
    SQLEvaluator,
)


class SpiderEvaluator(SQLEvaluator):
    """公开数据集SPIDER SQLite数据库评估"""

    def __init__(
        self,
        llm: LLM,
        embed_model: BaseEmbedding,
        database_folder_path: str,
        eval_file_path: str,
        analysis_config: Dict,
        history_file_path: Optional[str] = None,
        using_all_history: bool = True,
    ):
        self._llm = llm
        self._embed_model = embed_model
        self._database_folder_path = database_folder_path
        self._sqlite_config_list: List[Dict[str, SqliteAnalysisConfig]] = []
        self._eval_file_path = eval_file_path

        if history_file_path:
            try:
                with open(history_file_path, "r") as f:
                    history_list = json.load(f)
                logger.info("History_file successfully loaded.")
            except FileNotFoundError:
                logger.warning(f"File not found: {history_file_path}.")
                raise
        else:
            history_list = []

        # process sqlite_config and q-sql history
        database_folder = Path(database_folder_path)
        for db_file_path in tqdm(database_folder.rglob("*")):
            if db_file_path.is_file():
                if db_file_path.suffix.lower() == ".sqlite":
                    # prepare sqlite_config for each database file
                    db_name = db_file_path.stem
                    db_path = str(db_file_path.parent)
                    sqlite_config = SqliteAnalysisConfig(
                        db_path=db_path,
                        database=db_name,
                        enable_enhanced_description=analysis_config.get(
                            "enable_enhanced_description", False
                        ),
                        enable_db_history=analysis_config.get(
                            "enable_db_history", False
                        ),
                        enable_db_embedding=analysis_config.get(
                            "enable_db_embedding", False
                        ),
                        max_col_num=analysis_config.get("max_col_num", 100),
                        max_val_num=analysis_config.get("max_val_num", 10000),
                        enable_query_preprocessor=analysis_config.get(
                            "enable_query_preprocessor", False
                        ),
                        enable_db_preretriever=analysis_config.get(
                            "enable_db_preretriever", False
                        ),
                        enable_db_selector=analysis_config.get(
                            "enable_db_selector", False
                        ),
                    )
                    if using_all_history:
                        db_history_list = [
                            {"query": item["question"], "SQL": item["query"]}
                            for item in history_list
                        ]
                    else:
                        db_history_list = []
                        for history_item in history_list:
                            if db_name == history_item["db_id"]:
                                db_history_list.append(
                                    {
                                        "query": history_item["question"],
                                        "SQL": history_item["query"],
                                    }
                                )

                    # connect each database and load its info
                    db_connector = SqliteConnector(sqlite_config)
                    sql_databse = db_connector.connect()
                    self._sqlite_config_list.append(
                        {
                            "db_name": db_name,
                            "sqlite_config": sqlite_config,
                            "sql_database": sql_databse,
                            "query_history": db_history_list,
                        }
                    )

    async def abatch_loader(
        self,
    ):
        tasks = []
        for config_item in self._sqlite_config_list:
            schema_retriever = resolve_schema_retriever(
                config_item["sqlite_config"], self._embed_model
            )
            history_retriever = resolve_history_retriever(
                config_item["sqlite_config"], self._embed_model
            )
            value_retriever = resolve_value_retriever(
                config_item["sqlite_config"], self._embed_model
            )
            db_loader = DBLoader(
                db_config=config_item["sqlite_config"],
                sql_database=config_item["sql_database"],
                embed_model=self._embed_model,
                llm=self._llm,
                schema_retriever=schema_retriever,
                history_retriever=history_retriever,
                value_retriever=value_retriever,
            )
            # 创建任务列表
            tasks.append(db_loader.aload_db_info(config_item["query_history"]))

        # 并发运行所有任务
        await asyncio.gather(*tasks)

        # # 控制并发任务数量
        # semaphore = asyncio.Semaphore(20)

        # async def limited_load(config_item):
        #     async with semaphore:
        #         try:
        #             await self._aload_config_item(config_item)
        #             logger.info(
        #                 f"""Successfully loaded database {config_item["sql_database"]}"""
        #             )
        #         except Exception as e:
        #             logger.error(
        #                 f"""Error loaded database {config_item["sql_database"]}: {e}"""
        #             )

        # tasks = [limited_load(config_item) for config_item in self._sqlite_config_list]

        # await asyncio.gather(*tasks)

    # async def _aload_config_item(self, config_item):
    #     for config_item in self._sqlite_config_list:
    #         schema_retriever = resolve_schema_retriever(config_item, self._embed_model)
    #         value_retriever = resolve_value_retriever(config_item, self._embed_model)
    #         db_loader = DBLoader(
    #             db_config=config_item["sqlite_config"],
    #             sql_database=config_item["sql_database"],
    #             embed_model=self._embed_model,
    #             llm=self._llm,
    #             schema_retriever=schema_retriever,
    #             value_retriever=value_retriever,
    #         )
    #         await db_loader.aload_db_info()

    async def abatch_query(self, nums: int):
        with open(self._eval_file_path, "r") as f:
            eval_list = json.load(f)

        # 控制并发任务数量
        semaphore = asyncio.Semaphore(20)

        async def limited_query(index, eval_item):
            async with semaphore:
                try:
                    result = await self._aquery_eval_item(index, eval_item)
                    logger.info(f"Successfully processed query {index}")
                    return result
                except Exception as e:
                    logger.error(f"Error processing query {index}: {e}")
                    return (index, None, None)  # 返回默认值或特殊标记

        tasks = [
            limited_query(i, eval_item) for i, eval_item in enumerate(eval_list[0:nums])
        ]

        # # 创建带有索引的任务列表
        # tasks = [
        #     self._aquery_eval_item(i, eval_item)
        #     for i, eval_item in enumerate(eval_list[0:nums])
        # ]

        # 并发运行所有任务并收集带索引的结果
        results = await asyncio.gather(*tasks)

        # 按索引排序结果
        sorted_results = sorted(results, key=lambda x: x[0])

        # 分离预测的 SQL 列表和查询结果列表
        predicted_sql_list = [item[1] for item in sorted_results]
        queried_result_list = [item[2] for item in sorted_results]

        return predicted_sql_list, queried_result_list

    async def _aquery_eval_item(self, idx, eval_item):
        for config_item in self._sqlite_config_list:
            if eval_item["db_id"] == config_item["db_name"]:
                schema_retriever = resolve_schema_retriever(
                    config_item["sqlite_config"], self._embed_model
                )
                history_retriever = resolve_history_retriever(
                    config_item["sqlite_config"], self._embed_model
                )
                value_retriever = resolve_value_retriever(
                    config_item["sqlite_config"], self._embed_model
                )
                db_query = DBQuery(
                    db_config=config_item["sqlite_config"],
                    sql_database=config_item["sql_database"],
                    embed_model=self._embed_model,
                    llm=self._llm,
                    schema_retriever=schema_retriever,
                    history_retriever=history_retriever,
                    value_retriever=value_retriever,
                )
                query_bundle = QueryBundle(eval_item["question"])

                response_node, metadata = await db_query.aquery_pipeline(query_bundle)

                return (
                    idx,
                    response_node[0].metadata["query_code_instruction"],
                    response_node[0].metadata["query_output"],
                )

    def batch_query(self, nums: int):
        with open(self._eval_file_path, "r") as f:
            eval_list = json.load(f)

        predicted_sql_list = []
        queried_result_list = []
        for eval_item in eval_list[0:nums]:
            for config_item in self._sqlite_config_list:
                if eval_item["db_id"] == config_item["db_name"]:
                    schema_retriever = resolve_schema_retriever(
                        config_item["sqlite_config"], self._embed_model
                    )
                    history_retriever = resolve_history_retriever(
                        config_item["sqlite_config"], self._embed_model
                    )
                    value_retriever = resolve_value_retriever(
                        config_item["sqlite_config"], self._embed_model
                    )
                    db_query = DBQuery(
                        db_config=config_item["sqlite_config"],
                        sql_database=config_item["sql_database"],
                        embed_model=self._embed_model,
                        llm=self._llm,
                        schema_retriever=schema_retriever,
                        history_retriever=history_retriever,
                        value_retriever=value_retriever,
                    )
                    query_bundle = QueryBundle(eval_item["question"])

                    response_node, metadata = db_query.query_pipeline(query_bundle)
                    predicted_sql_list.append(
                        response_node[0].metadata["query_code_instruction"]
                    )
                    queried_result_list.append(
                        response_node[0].metadata["query_output"]
                    )

        return predicted_sql_list, queried_result_list

    def parse_predicted_sql(self, predicted_sql_str: str):
        if "\n" in predicted_sql_str:
            return predicted_sql_str.replace("\n", " ")
        else:
            return predicted_sql_str

    def batch_evaluate(
        self, gold_file: str, predicted_file: str, evaluation_type: str, table_file: str
    ):
        kmaps = build_foreign_key_map_from_json(table_file)

        evaluate(
            gold_file,
            predicted_file,
            self._database_folder_path,
            evaluation_type,
            kmaps,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
