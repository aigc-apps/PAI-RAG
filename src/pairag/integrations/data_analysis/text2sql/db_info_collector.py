from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, cast
import os
import json
from decimal import Decimal
import datetime
from loguru import logger
from sqlalchemy import Table
from pathlib import Path
import pandas as pd

from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.objects.table_node_mapping import SQLTableSchema

from pairag.integrations.data_analysis.text2sql.utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
    DEFAULT_DB_DESCRIPTION_NAME,
    DEFAULT_DESCRIPTION_FOLDER_PATH,
    DEFAULT_TABLE_COMMENT_PATH,
    DEFAULT_TABLE_COMMENT_NAME,
    DEFAULT_DB_HISTORY_PATH,
    DEFAULT_DB_HISTORY_NAME,
    DEFAULT_DB_VALUE_PATH,
    DEFAULT_DB_VALUE_NAME,
)


class DBInfoCollector(ABC):
    """数据库信息处理接口"""

    @abstractmethod
    def collect(self):
        pass


# schema信息处理接口实现
class SchemaCollector(DBInfoCollector):
    def __init__(
        self,
        db_name: str,
        sql_database: SQLDatabase,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
        table_comment_file_path: Optional[str] = None,
        db_description_file_path: Optional[str] = None,
    ):
        self._db_name = db_name
        self._sql_database = sql_database
        self._tables = list(sql_database._usable_tables)
        self._dialect = sql_database.dialect
        self._table_schema_objs = self._load_get_tables_fn(
            sql_database, self._tables, context_query_kwargs, table_retriever
        )
        table_comment_file_path = table_comment_file_path or DEFAULT_TABLE_COMMENT_PATH
        self._table_comment_file_path = os.path.join(
            table_comment_file_path, f"{db_name}_{DEFAULT_TABLE_COMMENT_NAME}"
        )
        db_description_file_path = (
            db_description_file_path or DEFAULT_DB_DESCRIPTION_PATH
        )
        self._db_description_file_path = os.path.join(
            db_description_file_path, f"{db_name}_{DEFAULT_DB_DESCRIPTION_NAME}"
        )

    def collect(self):
        structured_db_description_dict = self._get_structured_db_description()
        logger.info(f"Structured_db_description obtained for {self._db_name}.")
        # 保存为json文件
        save_to_json(structured_db_description_dict, self._db_description_file_path)
        logger.info(
            f"structured_db_description saved to: {self._db_description_file_path}"
        )

        return structured_db_description_dict

    def _get_structured_db_description(self) -> Dict:
        """
        Get structured schema with data samples from database + optional context description from webui
        """
        table_info_list = []
        table_foreign_key_list = []

        for (
            table_schema_obj
        ) in (
            self._table_schema_objs
        ):  # a list of SQLTableSchema, e.g. [SQLTableSchema(table_name='has_pet', context_str=None),]
            table_name = table_schema_obj.table_name
            try:
                comment_from_db = self._sql_database._inspector.get_table_comment(
                    table_name, schema=self._sql_database._schema
                )["text"]
            except NotImplementedError as e:
                logger.warning(f"dialect does not support comments: {e}")
                comment_from_db = self._get_table_comment_from_file(table_name)
            except Exception as e:
                logger.warning(f"Failed to get table comment: {e}")
                comment_from_db = None
            additional_desc = table_schema_obj.context_str
            logger.info(f"Additional comment: {additional_desc}")
            # get table description
            table_comment = self._merge_comment_and_desc(
                comment_from_db, additional_desc
            )
            # get table data samples
            data_sample = self._get_data_sample(table_name)
            # get table primary key
            table_pks = self._get_table_primary_key(table_name)
            # get foreign keys
            table_fks = self._get_table_foreign_keys(table_name)
            table_foreign_key_list.extend(table_fks)
            # get column info
            column_info_list = []
            for i, col in enumerate(
                self._sql_database._inspector.get_columns(
                    table_name, self._sql_database._schema
                )
            ):
                # print("col:", col, "data_sample:", data_sample)
                column_value_sample = [row[i] for row in data_sample]
                # collect and structure table schema with data samples
                column_info_list.append(
                    {
                        "column_name": col["name"],
                        "column_type": str(col["type"]),
                        "column_comment": col.get("comment"),
                        "primary_key": col["name"] in table_pks,
                        "foreign_key": False,
                        "foreign_key_referred_table": None,
                        "column_value_sample": column_value_sample,
                        "column_description": None,
                    }
                )

            table_info_list.append(
                {
                    "table_name": table_name,
                    "table_comment": table_comment,
                    "table_description": None,
                    "column_info": column_info_list,
                }
            )

        # 处理table之间的foreign key一致性
        table_info_list = self._keep_foreign_keys_consistency(
            table_foreign_key_list, table_info_list
        )

        structured_db_description_dict = {
            "db_overview": None,
            "table_info": table_info_list,
        }

        return structured_db_description_dict

    def _load_get_tables_fn(
        self,
        sql_database: SQLDatabase,
        tables: Optional[Union[List[str], List[Table]]] = None,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
    ) -> List[SQLTableSchema]:
        # Load get_tables function
        context_query_kwargs = context_query_kwargs or {}
        if table_retriever is not None:
            return lambda query_str: cast(Any, table_retriever).retrieve(query_str)
        else:
            if tables is not None:
                table_names: List[str] = [
                    t.name if isinstance(t, Table) else t for t in tables
                ]
            else:
                table_names = list(sql_database.get_usable_table_names())
            context_strs = [context_query_kwargs.get(t, None) for t in table_names]
            table_schemas = [
                SQLTableSchema(table_name=t, context_str=c)
                for t, c in zip(table_names, context_strs)
            ]
            return table_schemas

    def _get_table_comment_from_file(self, table_name: str):
        """从文件读取table注释, 用于不支持内置comment的数据库，如sqlite"""
        comment_dict = self._get_comment_from_file()

        return comment_dict.get(table_name, None)

    def _get_comment_from_file(
        self,
    ) -> str:
        """从文件中读取table注释, 默认传入的格式为 [{"table": "table_name", "comment": "table_comment"}, ...]"""
        try:
            with open(self._table_comment_file_path, "r") as f:
                table_comment_list = json.load(f)
            comment_dict = {}
            for item in table_comment_list:
                comment_dict[item["table"]] = item["comment"]
        except Exception as e:
            logger.debug(
                f"Error loading table comment from {self._table_comment_file_path}: {e}"
            )
            comment_dict = {}

        return comment_dict

    def _get_data_sample(self, table: str, sample_n: int = 3, seed: int = 2024) -> List:
        """对table随机采样"""
        if self._dialect == "mysql":
            # MySQL 使用 RAND(seed) 函数
            sql_str = f"SELECT * FROM {table} ORDER BY RAND({seed}) LIMIT {sample_n};"
        elif self._dialect == "sqlite":
            # SQLite 可以使用 RANDOM() 函数，但没有直接的种子设置
            sql_str = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {sample_n};"
        # elif self._dialect == "postgresql":
        #     # PostgreSQL 可以使用 SETSEED() 函数设置随机种子
        #     set_seed_query = f"SELECT setseed({seed});"
        #     # table_sample, _ = self._sql_database.run_sql(set_seed_query)
        #     sql_str = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {sample_n};"

        table_sample, _ = self._sql_database.run_sql(sql_str)

        # 转换 Decimal 对象为 float，datetime 对象为字符串
        converted_table_sample = self._convert_data_sample_format(eval(table_sample))
        # print("converted_table_sample:", converted_table_sample)

        return converted_table_sample

    def _get_table_primary_key(self, table_name: str) -> List:
        table_pk_constraint = self._sql_database._inspector.get_pk_constraint(
            table_name, self._sql_database._schema
        )  # get primary key
        # if table_pk_constraint["constrained_columns"]:
        #     table_pks = table_pk_constraint["constrained_columns"]
        # else:
        #     table_pks = None

        return table_pk_constraint["constrained_columns"]

    def _get_table_foreign_keys(self, table_name: str) -> List:
        table_fks = []
        for foreign_key in self._sql_database._inspector.get_foreign_keys(
            table_name, self._sql_database._schema
        ):
            table_foreign_key = {
                "table_name": table_name,
                "column_name": foreign_key["constrained_columns"][0],
                "foreign_key": True,
                "foreign_key_referred_table": foreign_key["referred_table"],
            }
            table_fks.append(table_foreign_key)

        return table_fks

    def _merge_comment_and_desc(self, comment_from_db: str, additional_desc: str):
        target_comment = [
            value for value in [comment_from_db, additional_desc] if value is not None
        ]
        if len(target_comment) > 0:
            return ", ".join(target_comment)
        else:
            return None

    def _keep_foreign_keys_consistency(self, table_foreign_key_list, table_info_list):
        # 处理table之间的foreign key一致性
        for table_foreign_key in table_foreign_key_list:
            for table_item in table_info_list:
                for column_item in table_item["column_info"]:
                    if (
                        table_item["table_name"] == table_foreign_key["table_name"]
                        and column_item["column_name"]
                        == table_foreign_key["column_name"]
                    ):
                        column_item.update(
                            {
                                "foreign_key": True,
                                "foreign_key_referred_table": table_foreign_key[
                                    "foreign_key_referred_table"
                                ],
                            }
                        )
                    if (
                        table_item["table_name"]
                        == table_foreign_key["foreign_key_referred_table"]
                        and column_item["column_name"]
                        == table_foreign_key["column_name"]
                    ):
                        column_item.update(
                            {
                                "foreign_key": True,
                                "foreign_key_referred_table": table_foreign_key[
                                    "table_name"
                                ],
                            }
                        )
        return table_info_list

    def _convert_data_sample_format(self, data):
        """递归地将数据中的特殊类型转换为常规类型"""
        if isinstance(data, list):
            return [self._convert_data_sample_format(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._convert_data_sample_format(item) for item in data)
        elif isinstance(data, dict):
            return {
                key: self._convert_data_sample_format(value)
                for key, value in data.items()
            }
        elif isinstance(data, Decimal):
            return float(data)
        elif isinstance(data, datetime.datetime):
            return data.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(data, datetime.date):
            return data.strftime("%Y-%m-%d")
        elif isinstance(data, datetime.time):
            return data.strftime("%H:%M:%S")
        elif isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")  # 将 bytes 转换为 str
        else:
            return data


# history信息处理接口实现
class HistoryCollector(DBInfoCollector):
    def __init__(self, db_name: str, query_history_file_path: Optional[str] = None):
        self._db_name = db_name
        query_history_file_path = query_history_file_path or DEFAULT_DB_HISTORY_PATH
        self._query_history_file_path = os.path.join(
            query_history_file_path, f"{db_name}_{DEFAULT_DB_HISTORY_NAME}"
        )

    def collect(self, history_list: Optional[List] = None):
        if history_list:
            return history_list
        else:
            try:
                with open(self._query_history_file_path, "r") as f:
                    query_history_list = json.load(f)
                    logger.info(f"Q-SQL pair obtained for {self._db_name}")
                if not self._valid_check(query_history_list):
                    logger.warning("Invalid query history format")
                    query_history_list = []
            except Exception as e:
                logger.warning(
                    f"Error loading query history from {self._query_history_file_path}: {e}"
                )
                query_history_list = []

        return query_history_list

    def _valid_check(self, query_history_list: List):
        required_keys = {"query", "SQL"}
        errors = []

        # 检查是否为列表
        if not isinstance(query_history_list, List):
            errors.append(f"Expected a list, got {type(query_history_list)}.")
        for i, item in enumerate(query_history_list):
            # 检查是否为字典
            if not isinstance(item, Dict):
                errors.append(f"Expected a dictionary at item {i}, got {type(item)}.")
                continue
            # # 检查所有必需的键是否存在
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                errors.append(f"Missing keys: {missing_keys} in item {i}.")
                continue
            for key in required_keys:
                if not isinstance(item[key], str) or not item[key].strip():
                    errors.append(f"Item {i}, {key} is not a non-empty string.")
                break

        if errors:
            for error in errors:
                logger.warning(f"{error}")
            return False

        logger.info("Query history format is valid.")
        return True


# value信息处理接口实现
class ValueCollector(DBInfoCollector):
    def __init__(
        self,
        db_name: str,
        sql_database: SQLDatabase,
        db_description_file_path: Optional[str] = None,
        db_value_file_path: Optional[str] = None,
        max_col_num: int = 100,
        max_val_num: int = 10000,
    ):
        self._db_name = db_name
        self._sql_database = sql_database
        db_description_file_path = (
            db_description_file_path or DEFAULT_DB_DESCRIPTION_PATH
        )
        self._db_description_file_path = os.path.join(
            db_description_file_path, f"{db_name}_{DEFAULT_DB_DESCRIPTION_NAME}"
        )
        db_value_file_path = db_value_file_path or DEFAULT_DB_VALUE_PATH
        self._db_value_file_path = os.path.join(
            db_value_file_path, f"{db_name}_{DEFAULT_DB_VALUE_NAME}"
        )
        self._max_col_num = max_col_num
        self._max_val_num = max_val_num

    def collect(self):
        self._db_description_dict = self._load_db_description(
            self._db_description_file_path
        )
        unique_value_dict = self._get_unique_values(self._db_description_dict)
        logger.info(f"Unique_value_dict obtained for {self._db_name}.")
        # # 保存为json文件
        # save_to_json(unique_value_dict, self._db_value_file_path)
        # logger.info(f"unique_value_dict saved to: {self._db_value_file_path}")

        return unique_value_dict

    # def _get_unique_values(
    #     self,
    #     db_description_dict: Dict,
    # ) -> Dict[str, Dict[str, List[str]]]:
    #     """
    #     Retrieves unique text values from the database excluding primary keys.
    #     """
    #     unique_values: Dict[str, Dict[str, List[str]]] = {}
    #     column_count, value_count = 0, 0

    #     for table in db_description_dict["table_info"]:
    #         table_name = table["table_name"]
    #         # print("========table=====:", table_name)
    #         table_values: Dict[str, List[str]] = {}
    #         # 筛选是string类型但不是primary_key的column
    #         for column in table["column_info"]:
    #             if (column_count > self._max_col_num) or (value_count > self._max_val_num):
    #                 logger.warning(
    #                     f"Maximum limit reached, column_count is {column_count}, value_count is {value_count}."
    #                 )
    #                 break
    #             column_name = column["column_name"]
    #             column_type = column["column_type"]
    #             if (("VARCHAR" in column_type) and (column_type != "VARCHAR(1)")) or (
    #                 "TEXT" in column_type
    #             ):
    #                 if column["primary_key"]:
    #                     continue
    #                 if any(
    #                     keyword in column_name.lower()
    #                     for keyword in [
    #                         "_id",
    #                         " id",
    #                         "url",
    #                         "email",
    #                         "web",
    #                         "time",
    #                         "phone",
    #                         "date",
    #                         "address",
    #                     ]
    #                 ) or column_name.endswith("Id"):
    #                     continue

    #                 # 获取column数值的统计信息
    #                 sum_of_lengths, count_distinct = self._get_column_stats(
    #                     table_name, column_name
    #                 )
    #                 if sum_of_lengths is None or count_distinct == 0:
    #                     continue
    #                 average_length = round(sum_of_lengths / count_distinct, 3)
    #                 logger.debug(
    #                     f"Column: {column_name}, sum_of_lengths: {sum_of_lengths}, count_distinct: {count_distinct}, average_length: {average_length}"
    #                 )

    #                 # 根据统计信息筛选字段
    #                 if (
    #                     ("name" in column_name.lower() and sum_of_lengths < 5000000)
    #                     or (sum_of_lengths < 2000000 and average_length < 25)
    #                     or count_distinct < 100
    #                 ):
    #                     logger.debug(f"Fetching distinct values for {column_name}")
    #                     try:
    #                         fetched_values = self._sql_database.run_sql(
    #                             f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL"
    #                         )
    #                         fetched_values = fetched_values[1]["result"]
    #                         values = [str(value[0]) for value in fetched_values[0:self._max_val_num]]
    #                         column_count += 1
    #                         value_count += len(values)
    #                     except Exception:
    #                         values = []
    #                     logger.debug(f"Number of different values: {len(values)}")
    #                     table_values[column_name] = values

    #         unique_values[table_name] = table_values

    #     logger.info(f"column_count is {column_count}, value_count is {value_count}.")

    #     return unique_values

    def _get_unique_values(
        self,
        db_description_dict: Dict,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Retrieves unique text values from the database excluding primary keys.
        """
        unique_values: Dict[str, Dict[str, List[str]]] = {}
        total_column_count, total_value_count = 0, 0

        for table in db_description_dict["table_info"]:
            table_name = table["table_name"]
            (
                table_values,
                should_break,
                total_column_count,
                total_value_count,
            ) = self._process_table_values(table, total_column_count, total_value_count)
            unique_values[table_name] = table_values
            if should_break:
                break

        logger.info(
            f"Processed_column_count is {total_column_count}, processed_value_count is {total_value_count}."
        )

        return unique_values

    def _process_table_values(self, table, total_column_count, total_value_count):
        table_values: Dict[str, List[str]] = {}
        # 筛选是string类型但不是primary_key的column
        for column in table["column_info"]:
            if (total_column_count > self._max_col_num) or (
                total_value_count > self._max_val_num * self._max_col_num
            ):
                logger.warning(
                    f"""{self._db_name}, {table["table_name"]}: Maximum limit reached, column_count is {total_column_count}, value_count is {total_value_count}."""
                )
                return table_values, True, total_column_count, total_value_count
            column_name = column["column_name"]
            column_type = column["column_type"]
            if (("VARCHAR" in column_type) and (column_type != "VARCHAR(1)")) or (
                "TEXT" in column_type
            ):
                if column["primary_key"]:
                    continue
                if any(
                    keyword in column_name.lower()
                    for keyword in [
                        "_id",
                        " id",
                        "url",
                        "email",
                        "web",
                        "time",
                        "phone",
                        "date",
                        "address",
                    ]
                ) or column_name.endswith("Id"):
                    continue

                # 获取column数值的统计信息
                sum_of_lengths, count_distinct = self._get_column_stats(
                    table["table_name"], column_name
                )
                if sum_of_lengths is None or count_distinct == 0:
                    continue
                average_length = round(sum_of_lengths / count_distinct, 3)
                logger.debug(
                    f"Column: {column_name}, sum_of_lengths: {sum_of_lengths}, count_distinct: {count_distinct}, average_length: {average_length}"
                )

                # 根据统计信息筛选字段
                if (
                    ("name" in column_name.lower() and sum_of_lengths < 5000000)
                    or (sum_of_lengths < 2000000 and average_length < 25)
                    or count_distinct < 100
                ):
                    logger.debug(f"Fetching distinct values for {column_name}")
                    try:
                        fetched_values = self._sql_database.run_sql(
                            f"""SELECT DISTINCT `{column_name}` FROM `{table["table_name"]}` WHERE `{column_name}` IS NOT NULL"""
                        )
                        fetched_values = fetched_values[1]["result"]
                        values = [
                            str(value[0])
                            for value in fetched_values[0 : self._max_val_num]
                        ]
                        total_column_count += 1
                        total_value_count += len(values)
                    except Exception:
                        values = []
                    logger.debug(f"Number of different values: {len(values)}")
                    table_values[column_name] = values

        return table_values, False, total_column_count, total_value_count

    def _get_column_stats(self, table_name, column_name):
        try:
            result = self._sql_database.run_sql(
                f"""
                SELECT SUM(LENGTH(unique_values)), COUNT(unique_values)
                FROM (
                    SELECT DISTINCT `{column_name}` AS unique_values
                    FROM `{table_name}`
                    WHERE `{column_name}` IS NOT NULL
                ) AS subquery
            """
            )
            result = result[1]["result"][0]
        except Exception as e:
            logger.warning(f"no unique values found: {e}")
            result = 0, 0

        sum_of_lengths, count_distinct = result

        return sum_of_lengths, count_distinct

    def _load_db_description(self, db_description_file_path: str) -> Dict:
        if not os.path.exists(db_description_file_path):
            raise FileNotFoundError(f"File not found: {db_description_file_path}.")

        with open(db_description_file_path, "r") as f:
            db_description_dict = json.load(f)

        return db_description_dict

    # def _make_lsh(
    #     self,
    #     unique_values: Dict[str, Dict[str, List[str]]],
    #     signature_size: int,
    #     n_gram: int,
    #     threshold: float,
    #     verbose: bool = True,
    # ) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
    #     """
    #     Creates a MinHash LSH from unique values.

    #     Args:
    #         unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values.
    #         signature_size (int): The size of the MinHash signature.
    #         n_gram (int): The n-gram size for the MinHash.
    #         threshold (float): The threshold for the MinHash LSH.
    #         verbose (bool): Whether to display progress information.

    #     Returns:
    #         Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The MinHash LSH object and the dictionary of MinHashes.
    #     """
    #     lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
    #     minhashes: Dict[str, Tuple[MinHash, str, str, str]] = {}
    #     try:
    #         total_unique_values = sum(
    #             len(column_values)
    #             for table_values in unique_values.values()
    #             for column_values in table_values.values()
    #         )
    #         logger.info(f"Total unique values: {total_unique_values}")

    #         progress_bar = (
    #             tqdm(total=total_unique_values, desc="Creating LSH")
    #             if verbose
    #             else None
    #         )

    #         for table_name, table_values in unique_values.items():
    #             for column_name, column_values in table_values.items():
    #                 logger.info(
    #                     f"Processing {table_name}.{column_name} - {len(column_values)}"
    #                 )

    #                 for id, value in enumerate(column_values):
    #                     minhash = create_minhash(signature_size, value, n_gram)
    #                     minhash_key = f"{table_name}.{column_name}.{id}"
    #                     minhashes[minhash_key] = (
    #                         minhash,
    #                         table_name,
    #                         column_name,
    #                         value,
    #                     )
    #                     lsh.insert(minhash_key, minhash)

    #                     if verbose:
    #                         progress_bar.update(1)

    #         if verbose:
    #             progress_bar.close()
    #     except Exception as e:
    #         logger.error(f"Error creating LSH: {e}")

    #     return lsh, minhashes

    # def get_value_lsh(
    #     self,
    #     db_description_dict: Optional[Dict] = None,
    #     signature_size: int = 128,
    #     n_gram: int = 3,
    #     threshold: float = 0.2,
    #     verbose: bool = True,
    #     file_path: Optional[str] = None,
    # ) -> None:
    #     """
    #     Creates a MinHash LSH for the database and saves the results.
    #     """
    #     # get unique_values
    #     unique_values = self._get_unique_values(db_description_dict)
    #     # get lsh and minhashes
    #     lsh, minhashes = self._make_lsh(
    #         unique_values, signature_size, n_gram, threshold, verbose
    #     )

    #     if file_path is None:
    #         file_path = self._value_lsh_path
    #     try:
    #         # 检查文件夹是否存在
    #         if not os.path.exists(file_path):
    #             os.makedirs(file_path)

    #         with open(os.path.join(file_path, "lsh.pkl"), "wb") as file:
    #             pickle.dump(lsh, file)
    #         with open(os.path.join(file_path, "minhashes.pkl"), "wb") as file:
    #             pickle.dump(minhashes, file)
    #     except Exception as e:
    #         logger.error(f"Error saving lsh to file: {e}")


def save_to_json(content: Dict, file_path: str) -> None:
    """
    将内容保存到指定的文件中
    :param content: 要保存的内容
    :param file_path: 文件路径
    """
    try:
        directory = os.path.dirname(file_path)
        # 检查文件夹是否存在
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error saving content to file: {e}")
    return


# schema信息处理接口实现
class BirdSchemaCollector(DBInfoCollector):
    def __init__(
        self,
        db_name: str,
        sql_database: SQLDatabase,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
        table_comment_file_path: Optional[str] = None,
        database_file_path: Optional[str] = None,
    ):
        self._db_name = db_name
        self._sql_database = sql_database
        self._tables = list(sql_database._usable_tables)
        self._dialect = sql_database.dialect
        self._table_schema_objs = self._load_get_tables_fn(
            sql_database, self._tables, context_query_kwargs, table_retriever
        )
        table_comment_file_path = table_comment_file_path or DEFAULT_TABLE_COMMENT_PATH
        self._table_comment_file_path = os.path.join(
            table_comment_file_path, f"{db_name}_{DEFAULT_TABLE_COMMENT_NAME}"
        )
        self._db_description_save_path = os.path.join(
            DEFAULT_DB_DESCRIPTION_PATH, f"{db_name}_{DEFAULT_DB_DESCRIPTION_NAME}"
        )
        self._database_file_path = database_file_path or DEFAULT_DESCRIPTION_FOLDER_PATH

    def collect(self):
        structured_db_description_dict = self._get_structured_db_description()
        logger.info(f"Structured_db_description obtained for {self._db_name}.")
        # 保存为json文件
        save_to_json(structured_db_description_dict, self._db_description_save_path)
        logger.info(
            f"structured_db_description saved to: {self._db_description_save_path}"
        )

        return structured_db_description_dict

    def _get_structured_db_description(self) -> Dict:
        """
        Get structured schema with data samples from database + optional context description from webui
        """
        table_info_list = []
        table_foreign_key_list = []

        if self._database_file_path:
            desc_csv_files = self._load_db_desc_file()
        else:
            desc_csv_files = []

        for (
            table_schema_obj
        ) in (
            self._table_schema_objs
        ):  # a list of SQLTableSchema, e.g. [SQLTableSchema(table_name='has_pet', context_str=None),]
            table_name = table_schema_obj.table_name
            try:
                comment_from_db = self._sql_database._inspector.get_table_comment(
                    table_name, schema=self._sql_database._schema
                )["text"]
            except NotImplementedError as e:
                logger.debug(f"dialect does not support comments: {e}")
                comment_from_db = self._get_table_comment_from_file(table_name)
            except Exception as e:
                logger.warning(f"Failed to get table comment: {e}")
                comment_from_db = None
            additional_desc = table_schema_obj.context_str
            logger.debug(f"Additional comment: {additional_desc}")
            # get table description
            table_comment = self._merge_comment_and_desc(
                comment_from_db, additional_desc
            )
            # get table data samples
            data_sample = self._get_data_sample(table_name)
            # get table primary key
            table_pks = self._get_table_primary_key(table_name)
            # get foreign keys
            table_fks = self._get_table_foreign_keys(table_name)
            table_foreign_key_list.extend(table_fks)

            # get table description df
            table_desc_df = None
            if desc_csv_files:
                for file in desc_csv_files:
                    if file.stem == table_name:
                        try:
                            table_desc_df = pd.read_csv(file, encoding_errors="ignore")
                            break
                        except FileNotFoundError:
                            logger.error(f"Description file not found: {file}")
                            raise
                        except Exception as e:
                            logger.error(f"Failed to read {file}: {e}")
                            raise
                if table_desc_df is None:
                    logger.info(
                        f"No matching description file found for table: {table_name}"
                    )

            # get column info
            column_info_list = []
            for i, col in enumerate(
                self._sql_database._inspector.get_columns(
                    table_name, self._sql_database._schema
                )
            ):
                # print("col:", col, "data_sample:", data_sample)
                if isinstance(table_desc_df, pd.DataFrame):
                    df_row = table_desc_df[
                        table_desc_df["original_column_name"].str.strip()
                        == col["name"].strip()
                    ]
                    col_alias = (
                        f"""alias({df_row["column_name"].values[0]})"""
                        if not pd.isna(df_row["column_name"].values[0])
                        else None
                    )
                    col_desc = (
                        df_row["column_description"].values[0]
                        if not pd.isna(df_row["column_description"].values[0])
                        else None
                    )
                    value_desc = (
                        df_row["value_description"].values[0]
                        if not pd.isna(df_row["value_description"].values[0])
                        else None
                    )
                else:
                    col_alias, col_desc, value_desc = None, None, None

                col_comment = self._merge_comment_and_desc(col_alias, col_desc)

                column_value_sample = [row[i] for row in data_sample]

                # collect and structure table schema with data samples
                column_info_list.append(
                    {
                        "column_name": col["name"],
                        "column_type": str(col["type"]),
                        "column_comment": self._merge_comment_and_desc(
                            col_comment, col.get("comment")
                        ),
                        "primary_key": col["name"] in table_pks,
                        "foreign_key": False,
                        "foreign_key_referred_table": None,
                        "column_value_sample": column_value_sample,
                        "column_description": value_desc,
                    }
                )

            table_info_list.append(
                {
                    "table_name": table_name,
                    "table_comment": table_comment,
                    "table_description": None,
                    "column_info": column_info_list,
                }
            )

        # 处理table之间的foreign key一致性
        table_info_list = self._keep_foreign_keys_consistency(
            table_foreign_key_list, table_info_list
        )

        structured_db_description_dict = {
            "db_overview": None,
            "table_info": table_info_list,
        }

        return structured_db_description_dict

    def _load_get_tables_fn(
        self,
        sql_database: SQLDatabase,
        tables: Optional[Union[List[str], List[Table]]] = None,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
    ) -> List[SQLTableSchema]:
        # Load get_tables function
        context_query_kwargs = context_query_kwargs or {}
        if table_retriever is not None:
            return lambda query_str: cast(Any, table_retriever).retrieve(query_str)
        else:
            if tables is not None:
                table_names: List[str] = [
                    t.name if isinstance(t, Table) else t for t in tables
                ]
            else:
                table_names = list(sql_database.get_usable_table_names())
            context_strs = [context_query_kwargs.get(t, None) for t in table_names]
            table_schemas = [
                SQLTableSchema(table_name=t, context_str=c)
                for t, c in zip(table_names, context_strs)
            ]
            return table_schemas

    def _get_table_comment_from_file(self, table_name: str):
        """从文件读取table注释, 用于不支持内置comment的数据库，如sqlite"""
        comment_dict = self._get_comment_from_file()

        return comment_dict.get(table_name, None)

    def _get_comment_from_file(
        self,
    ) -> str:
        """从文件中读取table注释, 默认传入的格式为 [{"table": "table_name", "comment": "table_comment"}, ...]"""
        try:
            with open(self._table_comment_file_path, "r") as f:
                table_comment_list = json.load(f)
            comment_dict = {}
            for item in table_comment_list:
                comment_dict[item["table"]] = item["comment"]
        except Exception as e:
            logger.debug(
                f"Error loading table comment from {self._table_comment_file_path}: {e}"
            )
            comment_dict = {}

        return comment_dict

    def _load_db_desc_file(self):
        # 使用Path对象表示文件夹路径
        desc_folder_path = Path(
            os.path.join(
                self._database_file_path, self._db_name, "database_description"
            )
        )
        # 使用glob查找所有csv文件
        desc_csv_files = list(desc_folder_path.glob("*.csv"))

        return desc_csv_files

    def _get_data_sample(self, table: str, sample_n: int = 3, seed: int = 2024) -> List:
        """对table随机采样"""
        if self._dialect == "mysql":
            # MySQL 使用 RAND(seed) 函数
            sql_str = f"SELECT * FROM `{table}` ORDER BY RAND({seed}) LIMIT {sample_n};"
        elif self._dialect == "sqlite":
            # SQLite 可以使用 RANDOM() 函数，但没有直接的种子设置
            sql_str = f"""SELECT * FROM "{table}" ORDER BY RANDOM() LIMIT {sample_n};"""
        # elif self._dialect == "postgresql":
        #     # PostgreSQL 可以使用 SETSEED() 函数设置随机种子
        #     set_seed_query = f"SELECT setseed({seed});"
        #     # table_sample, _ = self._sql_database.run_sql(set_seed_query)
        #     sql_str = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {sample_n};"

        table_sample, _ = self._sql_database.run_sql(sql_str)

        # 转换 Decimal 对象为 float，datetime 对象为字符串
        converted_table_sample = self._convert_data_sample_format(eval(table_sample))
        # print("converted_table_sample:", converted_table_sample)

        return converted_table_sample

    def _get_table_primary_key(self, table_name: str) -> str:
        table_pk_constraint = self._sql_database._inspector.get_pk_constraint(
            table_name, self._sql_database._schema
        )

        return table_pk_constraint["constrained_columns"]

    def _get_table_foreign_keys(self, table_name: str) -> List:
        table_fks = []
        for foreign_key in self._sql_database._inspector.get_foreign_keys(
            table_name, self._sql_database._schema
        ):
            table_foreign_key = {
                "table_name": table_name,
                "column_name": foreign_key["constrained_columns"][0],
                "foreign_key": True,
                "foreign_key_referred_table": foreign_key["referred_table"],
            }
            table_fks.append(table_foreign_key)

        return table_fks

    def _merge_comment_and_desc(self, comment_from_db: str, additional_desc: str):
        target_comment = [
            value for value in [comment_from_db, additional_desc] if value is not None
        ]
        if len(target_comment) > 0:
            return ", ".join(target_comment)
        else:
            return None

    def _keep_foreign_keys_consistency(self, table_foreign_key_list, table_info_list):
        # 处理table之间的foreign key一致性
        for table_foreign_key in table_foreign_key_list:
            for table_item in table_info_list:
                for column_item in table_item["column_info"]:
                    if (
                        table_item["table_name"] == table_foreign_key["table_name"]
                        and column_item["column_name"]
                        == table_foreign_key["column_name"]
                    ):
                        column_item.update(
                            {
                                "foreign_key": True,
                                "foreign_key_referred_table": table_foreign_key[
                                    "foreign_key_referred_table"
                                ],
                            }
                        )
                    if (
                        table_item["table_name"]
                        == table_foreign_key["foreign_key_referred_table"]
                        and column_item["column_name"]
                        == table_foreign_key["column_name"]
                    ):
                        column_item.update(
                            {
                                "foreign_key": True,
                                "foreign_key_referred_table": table_foreign_key[
                                    "table_name"
                                ],
                            }
                        )
        return table_info_list

    def _convert_data_sample_format(self, data):
        """递归地将数据中的特殊类型转换为常规类型"""
        if isinstance(data, list):
            return [self._convert_data_sample_format(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._convert_data_sample_format(item) for item in data)
        elif isinstance(data, dict):
            return {
                key: self._convert_data_sample_format(value)
                for key, value in data.items()
            }
        elif isinstance(data, Decimal):
            return float(data)
        elif isinstance(data, datetime.datetime):
            return data.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(data, datetime.date):
            return data.strftime("%Y-%m-%d")
        elif isinstance(data, datetime.time):
            return data.strftime("%H:%M:%S")
        elif isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")  # 将 bytes 转换为 str
        else:
            return data
