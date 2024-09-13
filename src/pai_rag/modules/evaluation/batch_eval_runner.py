import asyncio
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index.core.async_utils import asyncio_module
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from pai_rag.integrations.evaluation.retrieval.evaluator import MyRetrievalEvalResult


async def eval_response_worker(
    semaphore: asyncio.Semaphore,
    evaluator: BaseEvaluator,
    evaluator_name: str,
    query: Optional[str] = None,
    response: Optional[Response] = None,
    reference: Optional[str] = None,
    eval_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, EvaluationResult]:
    """Get aevaluate_response tasks with semaphore."""
    eval_kwargs = eval_kwargs or {}
    async with semaphore:
        return (
            evaluator_name,
            await evaluator.aevaluate_response(
                query=query, response=response, reference=reference
            ),
        )


async def eval_retriever_worker(
    semaphore: asyncio.Semaphore,
    evaluator,
    evaluator_name: str,
    query: str,
    expected_ids: List[str],
) -> Tuple[str, MyRetrievalEvalResult]:
    async with semaphore:
        return (
            evaluator_name,
            await evaluator.aevaluate(query, expected_ids=expected_ids),
        )


async def response_worker(
    semaphore: asyncio.Semaphore,
    query_engine: BaseQueryEngine,
    query: str,
) -> RESPONSE_TYPE:
    """Get aquery tasks with semaphore."""
    async with semaphore:
        return await query_engine.aquery(query)


async def response_worker_for_retriever(
    semaphore: asyncio.Semaphore,
    retriever: BaseRetriever,
    query: str,
) -> RESPONSE_TYPE:
    """Get aquery tasks with semaphore."""
    async with semaphore:
        return await retriever.aretrieve(query)


class BatchEvalRunner:
    """Batch evaluation runner suitable for both retrieval and response modules.

    Args:
        retrieval_evaluators: Dictional of the retrieval evaluator.
        response_evaluators: Dictionary of response evaluators.
        workers (int): Number of workers to use for parallelization.
            Defaults to 2.
        show_progress (bool): Whether to show progress bars. Defaults to False.

    """

    def __init__(
        self,
        retrieval_evaluators: Dict[str, MyRetrievalEvalResult],
        response_evaluators: Dict[str, EvaluationResult],
        workers: int = 2,
        show_progress: bool = False,
    ):
        self.retrieval_evaluators = retrieval_evaluators
        self.response_evaluators = response_evaluators
        self.workers = workers
        self.semaphore = asyncio.Semaphore(self.workers)
        self.show_progress = show_progress
        self.asyncio_mod = asyncio_module(show_progress=self.show_progress)

    def _format_results(
        self, results: List[MyRetrievalEvalResult | EvaluationResult]
    ) -> Dict[str, List[MyRetrievalEvalResult | EvaluationResult]]:
        """Format results."""
        # Format results
        results_dict = {name: [] for name in self.retrieval_evaluators}
        results_dict_response = {name: [] for name in self.response_evaluators}
        results_dict.update(results_dict_response)
        for name, result in results:
            results_dict[name].append(result)
        return results_dict

    def _validate_and_clean_inputs(
        self,
        *inputs_list: Any,
    ) -> List[Any]:
        """Validate and clean input lists.

        Enforce that at least one of the inputs is not None.
        Make sure that all inputs have the same length.
        Make sure that None inputs are replaced with [None] * len(inputs).

        """
        assert len(inputs_list) > 0
        # first, make sure at least one of queries or response_strs is not None
        input_len: Optional[int] = None
        for inputs in inputs_list:
            if inputs is not None:
                input_len = len(inputs)
                break
        if input_len is None:
            raise ValueError("At least one item in inputs_list must be provided.")

        new_inputs_list = []
        for inputs in inputs_list:
            if inputs is None:
                new_inputs_list.append([None] * input_len)
            else:
                if len(inputs) != input_len:
                    raise ValueError("All inputs must have the same length.")
                new_inputs_list.append(inputs)
        return new_inputs_list

    def _get_eval_kwargs(
        self, eval_kwargs_lists: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """Get eval kwargs from eval_kwargs_lists at a given idx.

        Since eval_kwargs_lists is a dict of lists, we need to get the
        value at idx for each key.

        """
        return {k: v[idx] for k, v in eval_kwargs_lists.items()}

    async def aevaluate_responses(
        self,
        queries: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
        responses: Optional[List[Response]] = None,
        references: Optional[List[str]] = None,
        **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List]:
        """Evaluate query, response pairs.

        This evaluates queries and response objects.

        Args:
            queries (Optional[List[str]]): List of query strings. Defaults to None.
            responses (Optional[List[Response]]): List of response objects.
                Defaults to None.
            **eval_kwargs_lists (Dict[str, Any]): Dict of lists of kwargs to
                pass to evaluator. Defaults to None.

        """
        queries, responses = self._validate_and_clean_inputs(queries, responses)
        for k in eval_kwargs_lists:
            v = eval_kwargs_lists[k]
            if not isinstance(v, list):
                raise ValueError(
                    f"Each value in eval_kwargs must be a list. Got {k}: {v}"
                )
            eval_kwargs_lists[k] = self._validate_and_clean_inputs(v)[0]

        # run evaluations
        eval_jobs = []
        for idx, query in enumerate(cast(List[str], queries)):
            for name, evaluator in self.retrieval_evaluators.items():
                eval_jobs.append(
                    eval_retriever_worker(
                        self.semaphore,
                        evaluator,
                        name,
                        query=query,
                        expected_ids=node_ids[idx],
                    )
                )
            response = cast(List, responses)[idx]
            eval_kwargs = self._get_eval_kwargs(eval_kwargs_lists, idx)
            for name, evaluator in self.response_evaluators.items():
                eval_jobs.append(
                    eval_response_worker(
                        self.semaphore,
                        evaluator,
                        name,
                        query=query,
                        response=response,
                        reference=references[idx],
                        eval_kwargs=eval_kwargs,
                    )
                )
        results = await self.asyncio_mod.gather(*eval_jobs)

        # Format results
        return self._format_results(results)

    async def aevaluate_retrieval_res(
        self,
        queries: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
        responses: Optional[List[Response]] = None,
        **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List]:
        """Evaluate query, response pairs.

        This evaluates queries and response objects.

        Args:
            queries (Optional[List[str]]): List of query strings. Defaults to None.
            responses (Optional[List[Response]]): List of response objects.
                Defaults to None.
            **eval_kwargs_lists (Dict[str, Any]): Dict of lists of kwargs to
                pass to evaluator. Defaults to None.

        """
        queries, responses = self._validate_and_clean_inputs(queries, responses)
        for k in eval_kwargs_lists:
            v = eval_kwargs_lists[k]
            if not isinstance(v, list):
                raise ValueError(
                    f"Each value in eval_kwargs must be a list. Got {k}: {v}"
                )
            eval_kwargs_lists[k] = self._validate_and_clean_inputs(v)[0]

        # run evaluations
        eval_jobs = []
        for idx, query in enumerate(cast(List[str], queries)):
            for name, evaluator in self.retrieval_evaluators.items():
                eval_jobs.append(
                    eval_retriever_worker(
                        self.semaphore,
                        evaluator,
                        name,
                        query=query,
                        expected_ids=node_ids[idx],
                    )
                )
        results = await self.asyncio_mod.gather(*eval_jobs)

        # Format results
        return self._format_results(results)

    async def aevaluate_queries(
        self,
        query_engine,
        queries: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
        **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List]:
        """Evaluate queries.

        Args:
            query_engine (BaseQueryEngine): Query engine.
            queries (Optional[List[str]]): List of query strings. Defaults to None.
            **eval_kwargs_lists (Dict[str, Any]): Dict of lists of kwargs to
                pass to evaluator. Defaults to None.

        """
        if queries is None:
            raise ValueError("`queries` must be provided")

        # gather responses
        response_jobs = []
        for query in queries:
            response_jobs.append(response_worker(self.semaphore, query_engine, query))
        responses = await self.asyncio_mod.gather(*response_jobs)

        return await self.aevaluate_responses(
            queries=queries,
            node_ids=node_ids,
            responses=responses,
            references=reference_answers,
            **eval_kwargs_lists,
        )

    async def aevaluate_queries_for_retrieval(
        self,
        retriever,
        queries: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
        **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List]:
        """Evaluate queries.

        Args:
            query_engine (BaseQueryEngine): Query engine.
            queries (Optional[List[str]]): List of query strings. Defaults to None.
            **eval_kwargs_lists (Dict[str, Any]): Dict of lists of kwargs to
                pass to evaluator. Defaults to None.

        """
        if queries is None:
            raise ValueError("`queries` must be provided")

        # gather responses
        response_jobs = []
        for query in queries:
            response_jobs.append(
                response_worker_for_retriever(self.semaphore, retriever, query)
            )
        responses = await self.asyncio_mod.gather(*response_jobs)

        return await self.aevaluate_retrieval_res(
            queries=queries,
            node_ids=node_ids,
            responses=responses,
            **eval_kwargs_lists,
        )
