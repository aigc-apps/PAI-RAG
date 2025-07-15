import dotenv
dotenv.load_dotenv()

from celery import Celery
import os
from pairag.mcp.tools.knowledgebase.knowledgebase_tool import kb_client
from pairag.app.init import init_dependencies
import asyncio
from loguru import logger



DEFAULT_BROKER = "redis://localhost:6379/0"


app = Celery(
    "PAIRAG_WORKER",
    broker=os.environ.get("PAIRAG_BROKER") or DEFAULT_BROKER,
    backend=os.environ.get("PAIRAG_BROKER") or DEFAULT_BROKER,
)


@app.task(name="process_file")
def process_file(file_id: str):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_dependencies(init_mcp_tools=False))
    logger.info(f"Processing file {file_id}.")
    loop.run_until_complete(kb_client.process_file_async(file_id))
    logger.info(f"Processed file {file_id} successfully.")
