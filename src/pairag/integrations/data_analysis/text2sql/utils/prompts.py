from llama_index.core import PromptTemplate


DEFAULT_DB_SUMMARY_PROMPT = PromptTemplate(
    "下面是用户数据库{db_name}中数据表结构信息，抽样数据样例及描述:\n"
    "数据表结构信息和数据样例: {db_schema}, \n"
    "请学习理解该数据库的表结构和内容，按要求输出数据库整体描述，各数据表的描述以及各字段的描述。\n"
    "关于各字段描述的具体要求: \n"
    "1. 分析每个数据表中各列数据的含义和作用，并对专业术语进行简单明了的解释。\n"
    "2. 如果列数值是时间类型请给出时间格式，类似:yyyy-MM-dd HH:MM:ss或者yyyy-MM等。\n"
    "3. 请不要修改或者翻译列名，确保和给出数据列名一致。\n\n"
    "请一步一步思考，以中文回答。\n"
    "回答: "
)


# DEFAULT_KEYWORD_EXTRACTION_PROMPT = PromptTemplate(
#     "给定一个用户问题和提示，请仅基于该问题和提示，提取其中的关键词、关键短语和命名实体。这些元素对于理解问题的核心组成部分以及提供的指导至关重要。\n"
#     "要求: \n"
#     "1. **仔细阅读问题**：理解问题的主要焦点和具体细节。结合提示寻找任何命名实体（如组织、地点等）、技术术语和其他能概括问题重要方面的短语。\n"
#     "2. **关键词**：捕捉问题和提示本质方面的单个词语。**关键短语**：代表特定概念、地点、组织或其他重要细节的简短短语或命名实体。确保使用问题或提示中的原始措辞或术语。\n"
#     "3. **排除无关信息**：不要引入与用户问题和提示无关的外部信息或示例数据。\n"
#     "4. **将从问题中提取的关键词、关键短语和命名实体合并到一个列表中作为回答输出**，无需解释，回答样例：[关键词1, 关键词2,...]。\n"
#     "5. 提取的关键词、关键短语和命名实体的语言和用户问题保持一致。\n\n"
#     "参考示例: {fewshot_examples}\n"
#     "用户问题: {query_str}\n"
#     "提示: {hint}\n"
#     "回答: "
# )

DEFAULT_KEYWORD_EXTRACTION_PROMPT = PromptTemplate(
    """
Objective: Analyze the given question and hint to identify and extract keywords, keyphrases, and named entities.
These elements are crucial for understanding the core components of the inquiry and the guidance provided.
This process involves recognizing and isolating significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.\n\n
Instructions:\n
1. Read the Question Carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.\n
2. Analyze the Hint: The hint is designed to direct attention toward certain elements relevant to answering the question. Extract any keywords, phrases, or named entities that could provide further clarity or direction in formulating an answer.\n
3. List Keyphrases and Entities: Combine your findings from both the question and the hint into a single Python list. This list should contain:
- Keywords: Single words that capture essential aspects of the question or hint.
- Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.
Ensure to maintain the original phrasing or terminology used in the
question and hint.
4. The language of the listed Keywords, Keyphrases and Entities should be consistent with the question.\n\n
{fewshot_examples}\n\n
Task:\n
Given the following question and hint, identify and list all relevant keywords, keyphrases, and named entities.\n\n
Question: {query_str}\n
Hint: {hint}\n\n
Please provide your findings as a Python list, capturing the essence of both the question and hint through the identified terms and phrases.
Only output the Python list, no explanations needed.
"""
)


# DEFAULT_DB_SCHEMA_SELECT_PROMPT = PromptTemplate(
#     "以下是用户数据库信息描述: \n"
#     "数据表结构信息和数据样例: {db_schema} \n"
#     "请学习理解该数据的结构和内容, 根据用户问题和提示, 筛选出有用的表和列信息。\n"
#     "如有必要, 请将用户问题拆解, 一步步思考, 返回可能有用的数据表名和相关列名。\n\n"
#     "用户问题: {nl_query}\n"
#     "提示: {hint}\n"
#     "回答: \n"
# )

DEFAULT_DB_SCHEMA_SELECT_PROMPT = PromptTemplate(
    """
You are an expert and very smart data analyst.\n
Your task is to examine the provided database schema, understand the posed question, and use the hint to pinpoint the specific columns within tables that are essential for crafting a SQL query to answer the question.\n
Database Schema Overview: {db_schema}\n\n
This schema offers an in-depth description of the database’s architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints.
Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query.
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "– examples" in front of the corresponding column names.
This is a critical hint to identify the columns that will be used in the SQL query.\n\n
Question: {nl_query}\n
Hint: {hint}\n
The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.\n\n
Task:\n
Based on the database schema, question, and hint provided, your task is to identify all and only the columns that are essential for crafting a SQL query to answer the question.\n
For each of the selected columns, firstly think why exactly it is necessary for answering the question. Your reasoning should be concise and clear, demonstrating a logical connection between the columns and the question asked.\n
Tip: \n
If you are choosing a column for filtering a value within that column, make sure that column has the value as an example.
No need to output the reasoning for each column, only output the selected tables and columns.\n\n
"""
)


DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
    "Given a user question, please create a syntactically correct {dialect} query to execute according to the following requirements.\n"
    "Requirements:\n"
    "1. Fully understand the user's question and specific details based on the hint provided.\n"
    "2. Query only the relevant columns in specific tables based on the user's question and hint.\n"
    "3. Historical queries can be used as a reference to help generate the sql query.\n"
    "4. Use only column names from the provided database schema information. Do not query non-existent columns.\n"
    "5. Pay attention to which columns are in which tables. When necessary, qualify column names with their table names.\n"
    "6. If any table or column names in the generated SQL statement match SQL reserved words, such as ORDER, enclose these names in double quotes or backticks according to the {dialect}.\n\n"
    "User Question: {query_str}\n"
    "Hint: {hint}\n"
    "Database Schema Information and Sample Data: {db_schema}\n"
    "Historical Queries: {db_history}\n"
    "Take a deep breath and think step by step to find the correct SQL query but no need to output the chain of thought reasoning."
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query (end with ;) to run\n\n"
)
#     "给定一个用户问题，请按照以下要求创建一个语法正确的{dialect}查询语句来执行。\n"
#     "要求: \n"
#     "1. 请结合提示充分理解用户问题和具体细节。\n"
#     "1. 只根据用户问题和提示查询特定表中的相关列。\n"
#     "2. 请注意只使用提供数据库信息以及历史查询中看到的列名，不要查询不存在的列。\n"
#     "3. 请注意哪个列位于哪个表中。必要时，请使用表名限定列名。\n\n"
#     "4. 如生成SQL语句中有表名或列名与SQL保留字相同，如order，将该表名或列名加上双引号或反引号\n\n"
#     "用户问题: {query_str} \n"
#     "提示: {hint}\n"
#     "数据表结构信息和数据样例: {db_schema} \n"
#     "历史查询: {db_history} \n"
#     "You are required to use the following format, each taking one line:\n\n"
#     "Question: Question here\n"
#     "SQLQuery: SQL Query (end with ;) to run \n\n"
# )


DEFAULT_SQL_REVISION_PROMPT = PromptTemplate(
    "Given an input question, hint, database schema, sql execution result and query history, revise the predicted sql query following the correct {dialect} based on the instructions below.\n"
    "Instructions:\n"
    "1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is preferred over using MAX/MIN within sub queries.\n"
    "2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.\n"
    "3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.\n"
    "4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n"
    "5. Predicted query should return all of the information asked in the question without any missing or extra information.\n"
    "6. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, separated by a comma.\n"
    "7. When ORDER BY is used, just include the column name in the ORDER BY in the SELECT clause when explicitly asked in the question. Otherwise, do not include the column name in the SELECT clause.\n\n"
    "8. If execution result contains 'no such column', pay attention to which columns are in which tables from the provided schema information. Do not query non-existent columns.."
    "Question: {query_str}\n"
    "Hint: {hint}\n"
    "Database schema description: {db_schema}\n"
    "Query history: {db_history}\n"
    "Predicted sql query: {predicted_sql}\n"
    "SQL execution result: {sql_execution_result}\n"
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query (end with ;) to run \n\n"
)


DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    "给定一个输入问题，根据数据表信息描述、查询代码指令以及查询结果生成最终回复。\n"
    "要求: \n"
    "1.生成的回复语言需要与输入问题的语言保持一致。\n"
    "2.生成的回复需要关注数据表信息描述中可能存在的字段单位或其他补充信息。\n"
    "输入问题: {query_str} \n"
    "数据表信息描述: {db_schema} \n"
    "SQL 或 Python 查询代码指令（可选）: {query_code_instruction} \n"
    "查询结果: {query_output} \n\n"
    "最终回复: \n\n"
)
