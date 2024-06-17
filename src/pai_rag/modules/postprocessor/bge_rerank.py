import asyncio
from typing import Any, Coroutine, Dict, List, Optional
import multiprocessing as mp
from queue import Empty
from asgi_correlation_id import correlation_id
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from pai_rag.modules.postprocessor.base_rerank import CustomNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import datetime


def create_rerank_process(
    top_n: int = 5,
    model: str = "bge-reranker-base",
    use_fp16: bool = False,
    task_queue: mp.Queue = None,
    output_queue: mp.Queue = None,
):
    reranker = FlagEmbeddingReranker(top_n=top_n, model=model, use_fp16=use_fp16)

    print("Rerank model initialized. Waiting to receive rererank jobs.")
    while True:
        try:
            task_item = task_queue.get(block=True, timeout=20)

        except Empty:
            print(f"{datetime.datetime.now()} Empty queue, time out....")
            continue

        print(f"{datetime.datetime.now()} Start reranking")
        results = reranker.postprocess_nodes(
            task_item["nodes"], task_item["query_bundle"]
        )
        print(f"{datetime.datetime.now()} Finish reranking")

        output_queue.put((task_item["task_id"], results))


class MyBGEReranker(CustomNodePostprocessor):
    """BGE Reranker."""

    model: str = Field(description="BGE Reranker model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    use_fp16: bool = Field(description="Whether to use fp16 for inference.")
    _reranker: Any = PrivateAttr()
    _input_queue: mp.Queue = PrivateAttr()
    _output_queue: mp.Queue = PrivateAttr()
    _worker: mp.Process = PrivateAttr()
    _result_cache: Dict[str, Any] = PrivateAttr()

    def __init__(
        self, top_n: int = 5, model: str = "bge-reranker-base", use_fp16: bool = False
    ):
        super().__init__(top_n=top_n, model=model, use_fp16=use_fp16)
        ctx = mp.get_context("spawn")
        self._input_queue = ctx.Queue()
        self._output_queue = ctx.Queue()
        for i in range(3):
            _worker = ctx.Process(
                target=create_rerank_process,
                args=(
                    self.top_n,
                    self.model,
                    self.use_fp16,
                    self._input_queue,
                    self._output_queue,
                ),
            )
            _worker.start()

        self._result_cache = {}

    @classmethod
    def class_name(cls) -> str:
        return "MyBGEReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        task_id = correlation_id.get()
        work_item = {"task_id": task_id, "query_bundle": query_bundle, "nodes": nodes}
        self._input_queue.put(work_item)

        return []

    async def _postprocess_nodes_async(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> Coroutine[Any, Any, List[NodeWithScore]]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        task_id = correlation_id.get()
        work_item = {"task_id": task_id, "query_bundle": query_bundle, "nodes": nodes}
        self._input_queue.put(work_item)

        while True:
            if task_id in self._result_cache:
                results = self._result_cache[task_id]
                del self._result_cache[task_id]
                return results

            try:
                result_task_id, results = self._output_queue.get_nowait()
            except Empty:
                continue

            if result_task_id == task_id:
                return results
            else:
                self._result_cache[result_task_id] = results

            await asyncio.sleep(0.2)
