"""Used for schema description, q-sql history, and db value embedding"""
import os
import pickle
import hashlib
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Optional, Tuple
from datasketch import MinHash, MinHashLSH

from llama_index.core.schema import TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.utilities.sql_wrapper import SQLDatabase

from pai_rag.integrations.data_analysis.nl2sql.db_utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
    DEFAULT_DB_HISTORY_PATH,
    VALUE_LSH_PATH,
)
from pai_rag.integrations.data_analysis.nl2sql.db_utils.nl2sql_utils import (
    get_target_info,
    create_minhash,
)
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex


class DBIndexer:
    def __init__(
        self,
        sql_database: SQLDatabase,
        embed_model: BaseEmbedding,
        description_index: PaiVectorStoreIndex,
        history_index: PaiVectorStoreIndex,
        value_index: PaiVectorStoreIndex,
        db_description_path: Optional[str] = None,
        db_history_path: Optional[str] = None,
        # description_storage_path: Optional[str] = None,
        # history_storage_path: Optional[str] = None,
        # value_storage_path: Optional[str] = None,
        value_lsh_path: Optional[str] = None,
    ) -> None:
        self._sql_database = sql_database
        self._embed_model = embed_model
        self._db_description_path = db_description_path or DEFAULT_DB_DESCRIPTION_PATH
        self._db_history_path = db_history_path or DEFAULT_DB_HISTORY_PATH
        self._description_index = description_index
        self._history_index = history_index
        self._value_index = value_index

        # self._description_storage_path = (
        #     description_storage_path or os.path.join(DESCRIPTION_STORAGE_PATH, db_name)
        # )
        # self._history_storage_path = history_storage_path or os.path.join(HISTORY_STORAGE_PATH, db_name)
        # self._value_storage_path = value_storage_path or os.path.join(VALUE_STORAGE_PATH, db_name)
        self._value_lsh_path = value_lsh_path or VALUE_LSH_PATH

    def get_description_index(self, db_description_dict: Optional[Dict] = None):
        # get schema info
        db_description_dict = get_target_info(
            self._db_description_path, db_description_dict, flag="description"
        )
        # get nodes with embedding
        description_nodes = self._get_description_nodes_with_embedding(
            db_description_dict
        )
        # create & store index
        self._description_index.insert_nodes(description_nodes)
        # description_index = self._store_nodes_to_index(
        #     nodes=description_nodes, storage_path=self._description_storage_path
        # )  # 后续考虑根据db_name分别存储
        logger.info(
            f"DB Description Index created and stored. Number of nodes: {len(description_nodes)}"
        )

        # return description_index

    async def aget_description_index(self, db_description_dict: Optional[Dict] = None):
        # get schema info
        db_description_dict = get_target_info(
            self._db_description_path, db_description_dict, flag="description"
        )
        # get nodes with embedding
        description_nodes = await self._aget_description_nodes_with_embedding(
            db_description_dict
        )
        # create & store index
        self._description_index.insert_nodes(description_nodes)
        # description_index = self._store_nodes_to_index(
        #     nodes=description_nodes, storage_path=self._description_storage_path
        # )
        logger.info(
            f"DB Description Index created and stored. Number of nodes: {len(description_nodes)}"
        )

        # return description_index

    def get_history_index(self, db_history_list: Optional[List] = None):
        # get history info
        db_history_list = get_target_info(
            self._db_history_path, db_history_list, flag="history"
        )
        # get nodes with embedding
        history_nodes = self._get_history_nodes_with_embedding(db_history_list)
        # create & store index
        self._history_index.insert_nodes(history_nodes)
        # history_index = self._store_nodes_to_index(
        #     nodes=history_nodes, storage_path=self._history_storage_path
        # )
        logger.info(
            f"DB History Index created and stored. Number of nodes: {len(history_nodes)}"
        )

        # return history_index

    async def aget_history_index(self, db_history_list: Optional[List] = None):
        # get history info
        db_history_list = get_target_info(
            self._db_history_path, db_history_list, flag="history"
        )
        # get nodes with embedding
        history_nodes = await self._aget_history_nodes_with_embedding(db_history_list)
        # create & store index
        self._history_index.insert_nodes(history_nodes)
        # history_index = self._store_nodes_to_index(
        #     nodes=history_nodes, storage_path=self._history_storage_path
        # )
        logger.info(
            f"DB History Index created and stored. Number of nodes: {len(history_nodes)}"
        )

        # return history_index

    def get_value_index(
        self,
        db_description_dict: Optional[Dict] = None,
        max_col_num: int = 100,
        max_val_num: int = 10000,
    ):
        # get unique_values
        unique_values = self._get_unique_values(
            db_description_dict, max_col_num, max_val_num
        )
        logger.info(f"unique_values: {unique_values}")
        # get nodes with embedding
        value_nodes = self._get_value_nodes_with_embedding(unique_values)
        # create & store index
        self._value_index.insert_nodes(value_nodes)
        # value_index = self._store_nodes_to_index(
        #     nodes=value_nodes, storage_path=self._value_storage_path
        # )
        logger.info(
            f"DB Value Index created and stored. Number of nodes: {len(value_nodes)}"
        )

        # return value_index

    async def aget_value_index(
        self,
        db_description_dict: Optional[Dict] = None,
        max_col_num: int = 100,
        max_val_num: int = 10000,
    ):
        # get unique_values
        unique_values = self._get_unique_values(
            db_description_dict, max_col_num, max_val_num
        )
        logger.info(f"unique_values: {unique_values}")
        # get nodes with embedding
        value_nodes = await self._aget_value_nodes_with_embedding(unique_values)
        # create & store index
        self._value_index.insert_nodes(value_nodes)
        # value_index = self._store_nodes_to_index(
        #     nodes=value_nodes, storage_path=self._value_storage_path
        # )
        logger.info(
            f"DB Value Index created and stored. Number of nodes: {len(value_nodes)}"
        )

        # return value_index

    def _get_description_nodes_with_embedding(
        self, db_description_dict: Dict
    ) -> List[TextNode]:
        # get description nodes
        schema_description_nodes = self._get_nodes_from_db_description(
            db_description_dict
        )
        # get description nodes with embeddings
        schema_description_nodes = self._get_nodes_with_embeddings(
            schema_description_nodes
        )

        return schema_description_nodes

    async def _aget_description_nodes_with_embedding(
        self, db_description_dict: Dict
    ) -> List[TextNode]:
        # get description nodes
        schema_description_nodes = self._get_nodes_from_db_description(
            db_description_dict
        )
        # get description nodes with embeddings
        schema_description_nodes = await self._aget_nodes_with_embeddings(
            schema_description_nodes
        )

        return schema_description_nodes

    def _get_history_nodes_with_embedding(
        self, db_query_history: List
    ) -> List[TextNode]:
        # get history nodes
        query_history_nodes = self._get_nodes_from_db_history(db_query_history)
        # get history nodes with embeddings
        query_history_nodes = self._get_nodes_with_embeddings(query_history_nodes)

        return query_history_nodes

    async def _aget_history_nodes_with_embedding(
        self, db_query_history: List
    ) -> List[TextNode]:
        # get history nodes
        query_history_nodes = self._get_nodes_from_db_history(db_query_history)
        # get history nodes with embeddings
        query_history_nodes = await self._aget_nodes_with_embeddings(
            query_history_nodes
        )

        return query_history_nodes

    def _get_value_nodes_with_embedding(self, unique_values: Dict) -> List[TextNode]:
        # get unique values
        unique_value_nodes = self._get_nodes_from_db_values(unique_values)
        # get embeddings
        unique_value_nodes = self._get_nodes_with_embeddings(unique_value_nodes)

        return unique_value_nodes

    async def _aget_value_nodes_with_embedding(
        self, unique_values: Dict
    ) -> List[TextNode]:
        # get unique values
        unique_value_nodes = self._get_nodes_from_db_values(unique_values)
        # get embeddings
        unique_value_nodes = await self._aget_nodes_with_embeddings(unique_value_nodes)

        return unique_value_nodes

    def _get_nodes_with_embeddings(self, nodes: List[TextNode]):
        # get embeddings
        embeddings = self._embed_model.get_text_embedding_batch(
            [node.get_content(metadata_mode="embed") for node in nodes]
        )
        # update nodes embedding
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
            node.id_ = hashlib.sha256(node.text.encode()).hexdigest()

        return nodes

    async def _aget_nodes_with_embeddings(self, nodes: List[TextNode]):
        # get embeddings
        embeddings = await self._embed_model.aget_text_embedding_batch(
            [node.get_content(metadata_mode="embed") for node in nodes]
        )
        # update nodes embedding
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
            node.id_ = hashlib.sha256(node.text.encode()).hexdigest()

        return nodes

    def _get_unique_values(
        self,
        db_description_dict: Optional[Dict] = None,
        max_col_num: int = 100,
        max_val_num: int = 10000,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Retrieves unique text values from the database excluding primary keys.
        """

        # db_description_dict = self._get_schema_info(db_description_dict)
        db_description_dict = get_target_info(
            self._db_description_path, db_description_dict, "description"
        )
        unique_values: Dict[str, Dict[str, List[str]]] = {}
        column_count, value_count = 0, 0

        for table in db_description_dict["table_info"]:
            table_name = table["table_name"]
            # print("========table=====:", table_name)
            table_values: Dict[str, List[str]] = {}
            # 筛选是string类型但不是primary_key的column
            for column in table["column_info"]:
                if (column_count > max_col_num) or (value_count > max_val_num):
                    logger.warning(
                        f"Maximum limit reached, column_count is {column_count}, value_count is {value_count}."
                    )
                    break
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
                        table_name, column_name
                    )
                    if sum_of_lengths is None or count_distinct == 0:
                        continue
                    average_length = round(sum_of_lengths / count_distinct, 3)
                    logger.info(
                        f"Column: {column_name}, sum_of_lengths: {sum_of_lengths}, count_distinct: {count_distinct}, average_length: {average_length}"
                    )

                    # 根据统计信息筛选字段
                    if (
                        ("name" in column_name.lower() and sum_of_lengths < 5000000)
                        or (sum_of_lengths < 2000000 and average_length < 25)
                        or count_distinct < 100
                    ):
                        logger.info(f"Fetching distinct values for {column_name}")
                        try:
                            fetched_values = self._sql_database.run_sql(
                                f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL"
                            )
                            fetched_values = fetched_values[1]["result"]
                            values = [str(value[0]) for value in fetched_values]
                            column_count += 1
                            value_count += len(values)
                        except Exception:
                            values = []
                        logger.info(f"Number of different values: {len(values)}")
                        table_values[column_name] = values

            unique_values[table_name] = table_values

        logger.info(f"column_count is {column_count}, value_count is {value_count}.")

        return unique_values

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
            logger.info(f"no unique values found: {e}")
            result = 0, 0

        sum_of_lengths, count_distinct = result

        return sum_of_lengths, count_distinct

    # def _store_nodes_to_index(
    #     self, nodes: List[TextNode], storage_path: str
    # ) -> VectorStoreIndex:
    #     # create index
    #     index = VectorStoreIndex(nodes, embed_model=self._embed_model, storage_context=self._storage_context)
    #     # store index
    #     index.storage_context.persist(persist_dir=storage_path)

    #     return index

    def _get_nodes_from_db_description(
        self, db_description_dict: Dict
    ) -> List[TextNode]:
        schema_description_nodes = []
        for table in db_description_dict["table_info"]:
            table_desc = [
                value
                for value in [table["table_comment"], table["table_description"]]
                if value is not None
            ]
            if len(table_desc) > 0:
                table_desc = ", ".join(table_desc)
            else:
                table_desc = ""
            for column in table["column_info"]:
                column_desc = [
                    value
                    for value in [
                        column["column_comment"],
                        column["column_description"],
                    ]
                    if value is not None
                ]
                if len(column_desc) > 0:
                    column_desc = (
                        f"""{table["table_name"]} {column["column_name"]}: """
                        + ", ".join(column_desc)
                    )
                else:
                    column_desc = f"""{table["table_name"]} {column["column_name"]}"""

                metadata = {
                    "table_name": table["table_name"],
                    "column_name": column["column_name"],
                    "column_type": column["column_type"],
                    "table_description": table_desc,
                }
                schema_description_nodes.append(
                    TextNode(text=column_desc, metadata=metadata)
                )

        return schema_description_nodes

    def _get_nodes_from_db_history(self, db_query_history: List):
        query_history_nodes = []
        for item in db_query_history:
            query_history_nodes.append(
                TextNode(
                    text=item["query"],
                    metadata={"query": item["query"], "SQL": item["SQL"]},
                )
            )

        return query_history_nodes

    def _get_nodes_from_db_values(self, unique_values: Dict):
        unique_value_nodes = []
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                for value in column_values:
                    unique_value_nodes.append(
                        TextNode(
                            text=value,
                            metadata={
                                "table_name": table_name,
                                "column_name": column_name,
                            },
                        )
                    )

        return unique_value_nodes

    def _make_lsh(
        self,
        unique_values: Dict[str, Dict[str, List[str]]],
        signature_size: int,
        n_gram: int,
        threshold: float,
        verbose: bool = True,
    ) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
        """
        Creates a MinHash LSH from unique values.

        Args:
            unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values.
            signature_size (int): The size of the MinHash signature.
            n_gram (int): The n-gram size for the MinHash.
            threshold (float): The threshold for the MinHash LSH.
            verbose (bool): Whether to display progress information.

        Returns:
            Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The MinHash LSH object and the dictionary of MinHashes.
        """
        lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
        minhashes: Dict[str, Tuple[MinHash, str, str, str]] = {}
        try:
            total_unique_values = sum(
                len(column_values)
                for table_values in unique_values.values()
                for column_values in table_values.values()
            )
            logger.info(f"Total unique values: {total_unique_values}")

            progress_bar = (
                tqdm(total=total_unique_values, desc="Creating LSH")
                if verbose
                else None
            )

            for table_name, table_values in unique_values.items():
                for column_name, column_values in table_values.items():
                    logger.info(
                        f"Processing {table_name}.{column_name} - {len(column_values)}"
                    )

                    for id, value in enumerate(column_values):
                        minhash = create_minhash(signature_size, value, n_gram)
                        minhash_key = f"{table_name}.{column_name}.{id}"
                        minhashes[minhash_key] = (
                            minhash,
                            table_name,
                            column_name,
                            value,
                        )
                        lsh.insert(minhash_key, minhash)

                        if verbose:
                            progress_bar.update(1)

            if verbose:
                progress_bar.close()
        except Exception as e:
            logger.error(f"Error creating LSH: {e}")

        return lsh, minhashes

    def get_value_lsh(
        self,
        db_description_dict: Optional[Dict] = None,
        signature_size: int = 128,
        n_gram: int = 3,
        threshold: float = 0.2,
        verbose: bool = True,
        file_path: Optional[str] = None,
    ) -> None:
        """
        Creates a MinHash LSH for the database and saves the results.
        """
        # get unique_values
        unique_values = self._get_unique_values(db_description_dict)
        # get lsh and minhashes
        lsh, minhashes = self._make_lsh(
            unique_values, signature_size, n_gram, threshold, verbose
        )

        if file_path is None:
            file_path = self._value_lsh_path
        try:
            # 检查文件夹是否存在
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            with open(os.path.join(file_path, "lsh.pkl"), "wb") as file:
                pickle.dump(lsh, file)
            with open(os.path.join(file_path, "minhashes.pkl"), "wb") as file:
                pickle.dump(minhashes, file)
        except Exception as e:
            logger.error(f"Error saving lsh to file: {e}")
