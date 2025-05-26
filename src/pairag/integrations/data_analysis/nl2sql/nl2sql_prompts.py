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


DEFAULT_KEYWORD_EXTRACTION_PROMPT = PromptTemplate(
    "给定一个用户问题，请提取其中的关键词、关键短语和命名实体。这些元素对于理解问题的核心组成部分以及提供的指导至关重要。\n"
    "要求: \n"
    "1. 仔细阅读问题：理解问题的主要焦点和具体细节。寻找任何命名实体（如组织、地点等）、技术术语和其他能概括问题重要方面的短语。\n"
    "2. 关键词：捕捉问题或提示本质方面的单个词语。关键短语：代表特定概念、地点、组织或其他重要细节的简短短语或命名实体。确保使用问题中的原始措辞或术语。\n"
    # "3. 将从问题中提取的关键词、关键短语和命名实体合并到一个Python列表中，仅输出Python列表，无需解释。\n"
    "参考示例: {fewshot_examples}\n"
    "用户问题: {query_str}\n"
    "回答: "
)


DEFAULT_DB_SCHEMA_SELECT_PROMPT = PromptTemplate(
    "以下是用户数据库信息描述: \n"
    "数据表结构信息和数据样例: {db_schema} \n"
    "请学习理解该数据的结构和内容, 根据用户问题, 筛选出有用的表和列信息。\n"
    "如有必要, 请将用户问题拆解, 一步步思考, 返回可能有用的数据表名和相关列名。\n"
    "用户问题: {nl_query}\n"
    "回答: \n"
)


DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
    "给定一个用户问题，请按照以下要求创建一个语法正确的{dialect}查询语句来执行。\n"
    "要求: \n"
    "1. 只根据用户问题查询特定表中的相关列。\n"
    "2. 请注意只使用提供数据库信息以及可能历史查询中看到的列名，不要查询不存在的列。\n"
    "3. 请注意哪个列位于哪个表中。必要时，请使用表名限定列名。\n\n"
    "用户问题: {query_str} \n"
    "数据表结构信息和数据样例: {db_schema} \n"
    "历史查询: {db_history} \n"
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query (end with ;) to run \n\n"
)


DEFAULT_SQL_REVISION_PROMPT = PromptTemplate(
    "Given an input question, database schema, sql execution result and query history, revise the predicted sql query following the correct {dialect} based on the instructions below.\n"
    "Instructions:\n"
    "1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is preferred over using MAX/MIN within sub queries.\n"
    "2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.\n"
    "3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.\n"
    "4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n"
    "5. Predicted query should return all of the information asked in the question without any missing or extra information.\n"
    "6. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, separated by a comma.\n"
    "7. When ORDER BY is used, just include the column name in the ORDER BY in the SELECT clause when explicitly asked in the question. Otherwise, do not include the column name in the SELECT clause.\n\n"
    "Question: {query_str} \n"
    "Database schema description: {db_schema} \n"
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
