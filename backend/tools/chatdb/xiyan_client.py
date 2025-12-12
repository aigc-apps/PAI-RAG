import json
from common.llm.llm_model import PaiLlm
from loguru import logger
from xiyan_mcp_server.utils.db_config import DBConfig
from xiyan_mcp_server.utils.db_source import HITLSQLDatabase
from xiyan_mcp_server.utils.db_util import init_db_conn
from xiyan_mcp_server.utils.file_util import extract_sql_from_qwen
from xiyan_mcp_server.database_env import DataBaseEnv


class XiyanClient():
    def __init__(
        self,
        dialect: str,
        host: str,
        port: int,
        db_name: str,
        username: str,
        password: str,
        llm: PaiLlm,
    ):
        self.db_config = DBConfig(
            dialect=dialect,
            db_host=host,
            port=port,
            db_name=db_name,
            db_pwd=password,
            user_name=username,
        )
        self.llm = llm
        self.db_env = None

    def get_db_env(self):
        if self.db_env is None:
            db_conn = init_db_conn(self.db_config)
            db_source = HITLSQLDatabase(engine=db_conn)
            self.db_env = DataBaseEnv(db_source)

        return self.db_env


    async def execute_async(self, query: str):
        db_env = self.get_db_env()
        prompt = f"""你现在是一名{db_env.dialect}数据分析专家，你的任务是根据参考的数据库schema和用户的问题，编写正确的SQL来回答用户的问题，生成的SQL用``sql 和```包围起来。
    【数据库schema】
    {db_env.mschema_str}

    【问题】
    {query}
    """
        logger.info(f"SQL generation prompt: {prompt}")

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"用户的问题是: {query}"},
        ]

        try:
            sql_gen = await self.llm.astream(
                messages=messages
            )
            llm_content = ""
            async for chunk in sql_gen:
                llm_content += chunk.delta

            sql_query = extract_sql_from_qwen(llm_content)
            logger.info(f"Generated sql query: {sql_query}.")

            execution_round = 1
            while execution_round < 3:
                logger.info(f"Executing sql round {execution_round}: {sql_query}")
                status, res = db_env.database.fetch(sql_query)
                if status:
                    logger.info(f"Execution sql success: {res}")
                    break

                sql_query = await self.sql_fix_async(
                    dialect=db_env.dialect,
                    mschema=db_env.mschema_str,
                    query=query,
                    sql_query=sql_query,
                    error_info=res
                )
                execution_round += 1

            sql_res = db_env.database.fetch_truncated(sql_query, max_rows=100)
            markdown_res = db_env.database.trunc_result_to_markdown(sql_res).strip()
            logger.info(f"SQL query: {sql_query}\n SQL result: {sql_res}\n")
            return json.dumps({
                "sql": sql_query,
                "result": markdown_res
            }, ensure_ascii=False)
        except Exception as ex:
            # 出错时回收资源？
            logger.info(f"Execute sql failed: {ex}.")
            return str(ex)
    async def sql_fix_async(
        self,
        dialect: str,
        mschema: str,
        query: str,
        sql_query: str,
        error_info: str
    ):
        system_prompt = """现在你是一个{dialect}数据分析专家，需要阅读一个客户的问题，参考的数据库schema，该问题对应的待检查SQL，以及执行该SQL时数据库返回的语法错误，请你仅针对其中的语法错误进行修复，输出修复后的SQL。
注意：
1、仅修复语法错误，不允许改变SQL的逻辑。
2、生成的SQL用```sql 和```包围起来。

【数据库schema】
{schema}
""".format(dialect=dialect, schema=mschema)

        user_prompt = """【问题】
{question}

【待检查SQL】
{sql}

【错误信息】
{sql_res}""".format(question=query, sql=sql_query, sql_res=error_info)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        sql_gen = await self.llm.astream(
            messages=messages
        )
        llm_content = ""
        async for chunk in sql_gen:
            llm_content += chunk.delta

        sql_query = extract_sql_from_qwen(llm_content)

        return sql_query
