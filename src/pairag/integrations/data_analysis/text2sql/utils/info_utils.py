from typing import Dict
import jieba
from datasketch import MinHash
from loguru import logger


def count_total_columns(db_description_dict: Dict) -> int:
    if len(db_description_dict) == 0:
        raise ValueError("db_description_dict is Empty")
    total_columns = 0
    for table in db_description_dict["table_info"]:
        total_columns += len(table["column_info"])

    return total_columns


def get_schema_desc(db_description_dict: Dict) -> str:
    """get schema description for llm"""
    if len(db_description_dict) == 0:
        raise ValueError("db_description_dict is Empty")
    # 获取db_overview
    if db_description_dict["db_overview"]:
        schema_description_str = (
            f"""Database overview: {db_description_dict["db_overview"]}\n\n"""
        )
    else:
        schema_description_str = ""
    # 获取所有表的描述
    all_table_descriptions = []
    for item_table in db_description_dict["table_info"]:
        all_table_descriptions.append(_get_table_desc(item_table))
    # 拼接所有表的描述
    schema_description_str += "\n".join(all_table_descriptions)

    return schema_description_str


# def _get_table_desc(table_name: str, table_info_list: List) -> str:
def _get_table_desc(target_table_dict: Dict) -> str:
    """get single table description"""
    table_desc = f"""Table {target_table_dict["table_name"]} has columns: """
    for column in target_table_dict["column_info"]:
        table_desc += f"""{column["column_name"]} ({column["column_type"]})"""
        if column["primary_key"]:
            table_desc += ", primary key"
        if column["foreign_key"]:
            table_desc += f""", foreign key, referred table: {column["foreign_key_referred_table"]}"""
        table_desc += f""", with value sample: {column["column_value_sample"]}"""
        col_comment = [
            value
            for value in [column["column_comment"], column["column_description"]]
            if value is not None
        ]
        if len(col_comment) > 0:
            table_desc += f""", with description: {", ".join(col_comment)}; """
        else:
            table_desc += "; "
    table_comment = [
        value
        for value in [
            target_table_dict["table_comment"],
            target_table_dict["table_description"],
        ]
        if value is not None
    ]
    if len(table_comment) > 0:
        table_desc += f""" with table description: {", ".join(table_comment)}."""
    else:
        table_desc += "."

    return table_desc


# def get_target_info(
#     target_path: str,
#     target_file: Optional[Dict | List] = None,
#     flag: str = "description",
# ) -> Dict | List:
#     # 正常情况下接受传入的description dict 或 history list，否则从本地加载
#     if target_file is None:
#         if flag == "description":
#             if not os.path.exists(target_path):
#                 raise ValueError(
#                     f"db_description_file_path: {target_path} does not exist"
#                 )
#         if flag == "history":
#             if not os.path.exists(target_path):
#                 raise ValueError(f"db_history_file_path: {target_path} does not exist")
#         try:
#             with open(target_path, "r") as f:
#                 target_file = json.load(f)
#         except Exception as e:
#             # raise ValueError(f"Load target object from {file_path} failed: {e}")
#             if flag == "description":
#                 target_file = {}
#                 logger.error(f"Error loading db_description_dict: {e}")
#             if flag == "history":
#                 target_file = []
#                 logger.error(f"Error loading db_history_list: {e}")

#     return target_file


def extract_subset_from_description(
    retrieved_nodes_dict: Dict, db_description_dict: Dict
) -> Dict:
    if len(retrieved_nodes_dict) > 0:
        sub_db_description_dict = {
            "db_overview": db_description_dict["db_overview"],
            "table_info": [],
        }
        for table_item in db_description_dict["table_info"]:
            filter_columns = []
            for column_item in table_item["column_info"]:
                key = (table_item["table_name"], column_item["column_name"])
                # 筛选满足条件的列并更新value sample
                if key in retrieved_nodes_dict:
                    if len(retrieved_nodes_dict[key]) > 0:
                        column_item["column_value_sample"].extend(
                            retrieved_nodes_dict[key]
                        )
                        column_item["column_value_sample"] = list(
                            set(column_item["column_value_sample"])
                        )
                    filter_columns.append(column_item)
                # 保留主键和外键
                if ((column_item["primary_key"]) or (column_item["foreign_key"])) and (
                    column_item not in filter_columns
                ):
                    filter_columns.append(column_item)
            if len(filter_columns) > 0:
                sub_db_description_dict["table_info"].append(
                    {
                        "table_name": table_item["table_name"],
                        "table_comment": table_item["table_comment"],
                        "table_description": table_item["table_description"],
                        "column_info": filter_columns,
                    }
                )
        logger.debug(f"sub_db_description_dict: {sub_db_description_dict}")
        return sub_db_description_dict
    else:
        return db_description_dict


def is_chinese_char(char):
    """检查单个字符是否是中文字符"""
    return "\u4e00" <= char <= "\u9fff"


def is_chinese_string(value: str) -> bool:
    """检查字符串是否主要由中文字符组成"""
    if not value:
        return False
    chinese_count = sum(is_chinese_char(char) for char in value)
    return chinese_count / len(value) > 0.5


def is_english_string(value: str) -> bool:
    """检查字符串是否主要由英文字符组成"""
    if not value:
        return False
    english_count = sum(c.isascii() for c in value)
    return english_count / len(value) > 0.5


def create_minhash(signature_size: int, value: str, n_gram: int) -> MinHash:
    """
    Creates a MinHash object for a given string.

    Args:
        signature_size (int): The size of the MinHash signature.
        string (str): The input string to create the MinHash for.
        n_gram (int): The n-gram size for the MinHash.

    Returns:
        MinHash: The MinHash object for the input string.
    """

    m = MinHash(num_perm=signature_size)
    if not value:
        # Return a default MinHash object if the input string is empty
        return m
    if is_chinese_string(value):
        words = list(jieba.cut(value, cut_all=True))
        # 生成 n-gram
        n = n_gram
        words_gram = []
        while n > 1:
            grams_n = ["".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            words_gram.extend(grams_n)
            n -= 1
        words_gram.extend(words)
        for w in words_gram:
            m.update(w.encode("utf8"))
    elif is_english_string(value):
        for d in [value[i : i + n_gram] for i in range(len(value) - n_gram + 1)]:
            m.update(d.encode("utf8"))
    else:
        raise ValueError(
            "The input string contains characters that are neither Chinese nor English."
        )
    return m


def jaccard_similarity(m1: MinHash, m2: MinHash) -> float:
    """
    Computes the Jaccard similarity between two MinHash objects.

    Args:
        m1 (MinHash): The first MinHash object.
        m2 (MinHash): The second MinHash object.

    Returns:
        float: The Jaccard similarity between the two MinHash objects.
    """
    return m1.jaccard(m2)
