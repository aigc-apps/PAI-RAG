# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

from fastapi import FastAPI
from modules.LLMService import LLMService
import time
import os
from pydantic import BaseModel
import json
from args import parse_args
from modules.UI import *
from loguru import logger
import sys

# Disable unstructured analytics tracking to mitigate timeout issues during PAI-EAS service.
# This is achieved by setting an environment variable that the unstructured library recognizes.
# By doing this, we prevent the library's internal function `scarf_analytics()` from making
# network requests to "https://packages.unstructured.io", which was causing timeouts.

os.environ["SCARF_NO_ANALYTICS"] = "true"

_global_args = parse_args()
service = LLMService()

with open(_global_args.config) as f:
    _global_cfg = json.load(f)


class Query(BaseModel):
    question: str
    topk: int | None = None
    topp: float | None = 0.8
    temperature: float | None = 0.7
    vector_topk: int | None = 3
    score_threshold: float | None = 0.5


class LLMQuery(BaseModel):
    question: str
    topk: int | None = None
    topp: float | None = 0.8
    temperature: float | None = 0.7


class VectorQuery(BaseModel):
    question: str
    vector_topk: int | None = 3
    score_threshold: float | None = 0.5


app = FastAPI()


def setup_middleware(app):
    # reset current middleware to allow modifying user provided list
    app.middleware_stack = None
    configure_cors_middleware(app)
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly


def configure_cors_middleware(app):
    from fastapi.middleware.cors import CORSMiddleware

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }

    app.add_middleware(CORSMiddleware, **cors_options)


def add_general_url(app):
    setup_middleware(app)

    @app.post("/chat/llm")
    async def query_by_llm(query: LLMQuery):
        ans, lens, _ = service.query_only_llm(
            query=query.question,
            llm_topK=query.topk,
            llm_topp=query.topp,
            llm_temp=query.temperature,
        )
        return {"response": ans, "tokens": lens}

    @app.post("/chat/vectorstore")
    async def query_by_vectorstore(query: VectorQuery):
        ans, lens = service.query_only_vectorstore(
            query=query.question,
            topk=query.vector_topk,
            score_threshold=query.score_threshold,
        )
        return {"response": ans, "tokens": lens}

    @app.post("/chat/langchain")
    async def query_by_langchain(query: Query):
        ans, lens, _ = service.query_retrieval_llm(
            query=query.question,
            topk=query.vector_topk,
            score_threshold=query.score_threshold,
            llm_topK=query.topk,
            llm_topp=query.topp,
            llm_temp=query.temperature,
        )
        return {"response": ans, "tokens": lens}


def start_webui():
    global app

    logger.info("Starting Webui server...")
    ui = create_ui(service, _global_args, _global_cfg)
    # app = gr.mount_gradio_app(app, ui, path='')
    app, local_url, share_url = ui.queue(concurrency_count=1, max_size=64).launch(
        server_name="0.0.0.0",
        server_port=_global_args.port,
        prevent_thread_lock=True,
        # required in local env
        share=True
    )

    logger.info("Adding fast api url...")
    add_general_url(app)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level=_global_args.log_level)
    start_webui()
    while 1:
        time.sleep(0.01)