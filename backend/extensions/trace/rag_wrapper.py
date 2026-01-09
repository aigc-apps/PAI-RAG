from functools import wraps
import json
import os
from typing import List
from common.tool.search_result import SearchResult
from opentelemetry.trace.status import Status, StatusCode
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from extensions.trace.utils import pydantic_to_dict

from extensions.trace.tracer import get_tracer

GEN_AI_SPAN_KIND = "gen_ai.span.kind"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
INPUT_MESSAGES = "gen_ai.input.messages"

INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE

RETRIEVER_SPAN_KIND = OpenInferenceSpanKindValues.RETRIEVER.value
EMBEDDING_SPAN_KIND = OpenInferenceSpanKindValues.EMBEDDING.value
RERANKER_SPAN_KIND = OpenInferenceSpanKindValues.RERANKER.value

RETRIEVE_KNOWLEDGE_OPERATION_NAME = "retrieve_knowledge"
VECTOR_SEARCH_OPERATION_NAME = "vector_search"
TEXT_SEARCH_OPERATION_NAME = "text_search"
EMBEDDING_OPERATION_NAME = "embeddings"
RERANKER_OPERATION_NAME = "rerank"

EMBEDDING_MDOEL_NAME = "gen_ai.request.model"
EMBEDDING_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"

RERANKER_MODEL_NAME = "gen_ai.request.model"

# whether to disable legacy trace and only use agentscope data contract
DISABLE_LEGACY_TRACE = os.getenv("DISABLE_LEGACY_TRACE", "false").lower() in ["true", "1", "yes", "y"]


STATUS_OK = Status(StatusCode.OK)


def query_knowledgebase_wrapper(func):
    """decorator to capture input & output string of query knowledgebase operation."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(self, *args, **kwargs)

        query = kwargs.get("query", "[unknown]")
        messages = [{
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": query,
                }
            ],
            "metadata": kwargs,
        }]
        span_name = RETRIEVE_KNOWLEDGE_OPERATION_NAME
        kb_id = kwargs.get("kb_id", None)
        if kb_id:
            span_name = f"{span_name} {kb_id}"

        with get_tracer().start_as_current_span(span_name) as span:
            try:
                span.set_attribute(GEN_AI_SPAN_KIND, RETRIEVER_SPAN_KIND)
                span.set_attribute(INPUT_MESSAGES, json.dumps(pydantic_to_dict(messages), ensure_ascii=False))
                retrieval_setting = kwargs.get("retrieval_setting", None)
                input_data = {
                    "query": query,
                    "top_k": retrieval_setting.top_k if retrieval_setting else None,
                    "score_threshold": retrieval_setting.similarity_threshold if retrieval_setting else None,
                }
                span.set_attribute(INPUT_VALUE, json.dumps(input_data, ensure_ascii=False))
                span.set_attribute(GEN_AI_OPERATION_NAME, RETRIEVE_KNOWLEDGE_OPERATION_NAME)

                results: List[SearchResult] = await func(self, *args, **kwargs)
                output_documents = [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "score": doc.score,
                        "metadata": doc.metadata,
                    }
                    for doc in results
                ]
                output_data = {
                    "documents": output_documents,
                    "document_size": len(output_documents),
                }
                output_value = json.dumps(pydantic_to_dict(output_data), ensure_ascii=False)
                span.set_attribute(OUTPUT_VALUE, output_value)
                span.set_status(STATUS_OK)
                return results
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def text_search_wrapper(func):
    """decorator to capture input & output string of query vector store operation."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(*args, **kwargs)

        query = kwargs.get("query", "[unknown]")
        messages = [{
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": query,
                }
            ],
        }]

        kb_id = kwargs.get("kb_id", None)
        span_name = TEXT_SEARCH_OPERATION_NAME
        if kb_id:
            span_name = f"{span_name} {kb_id}"

        with get_tracer().start_as_current_span(span_name) as span:
            try:
                span.set_attribute(GEN_AI_SPAN_KIND, RETRIEVER_SPAN_KIND)
                span.set_attribute(INPUT_MESSAGES, json.dumps(pydantic_to_dict(messages), ensure_ascii=False))
                input_data = {
                    "query": query,
                    "top_k": kwargs.get("top_k", None),
                }
                span.set_attribute(INPUT_VALUE, json.dumps(input_data, ensure_ascii=False))

                span.set_attribute(GEN_AI_OPERATION_NAME, TEXT_SEARCH_OPERATION_NAME)

                text_search_result = await func(*args, **kwargs)
                output_documents =[]
                if text_search_result and text_search_result.nodes:
                    output_documents = [
                        {
                            "id": text_search_result.ids[i],
                            "content": text_search_result.nodes[i].text,
                            "score": text_search_result.similarities[i],
                            "metadata": text_search_result.nodes[i].metadata,
                        }
                        for i in range(len(text_search_result.nodes))
                    ]
                output_data = {
                    "documents": output_documents,
                    "document_size": len(output_documents),
                }
                output_value = json.dumps(pydantic_to_dict(output_data), ensure_ascii=False)
                span.set_attribute(OUTPUT_VALUE, output_value)
                span.set_status(STATUS_OK)
                return text_search_result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def vector_search_wrapper(func):
    """decorator to capture input & output string of query vector store operation."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(*args, **kwargs)

        query = kwargs.get("query", "[unknown]")
        messages = [{
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": query,
                }
            ],
        }]
        kb_id = kwargs.get("kb_id", None)
        span_name = VECTOR_SEARCH_OPERATION_NAME
        if kb_id:
            span_name = f"{span_name} {kb_id}"
        with get_tracer().start_as_current_span(span_name) as span:
            try:
                span.set_attribute(GEN_AI_SPAN_KIND, RETRIEVER_SPAN_KIND)
                span.set_attribute(INPUT_MESSAGES, json.dumps(pydantic_to_dict(messages), ensure_ascii=False))
                input_data = {
                    "query": query,
                    "top_k": kwargs.get("top_k", None),
                }
                span.set_attribute(INPUT_VALUE, json.dumps(input_data, ensure_ascii=False))
                span.set_attribute(GEN_AI_OPERATION_NAME, VECTOR_SEARCH_OPERATION_NAME)

                vector_search_result = await func(*args, **kwargs)
                output_documents =[]
                if vector_search_result and vector_search_result.nodes:
                    output_documents = [
                    {
                        "id": vector_search_result.ids[i],
                        "content": vector_search_result.nodes[i].text,
                        "score": vector_search_result.similarities[i],
                        "metadata": vector_search_result.nodes[i].metadata,
                    }
                    for i in range(len(vector_search_result.nodes))
                ]
                output_data = {
                    "documents": output_documents,
                    "document_size": len(output_documents),
                }
                output_value = json.dumps(pydantic_to_dict(output_data), ensure_ascii=False)
                span.set_attribute(OUTPUT_VALUE, output_value)
                span.set_status(STATUS_OK)
                return vector_search_result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def embedding_wrapper(func):
    """decorator to capture input & output string of query knowledgebase operation."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(self, *args, **kwargs)

        query = kwargs.get("query", "[unknown]")
        embedding_model_entity = kwargs.get("embedding_model_entity", None)
        messages = [{
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": query,
                }
            ],
        }]
        try:
            span_name = f"{EMBEDDING_OPERATION_NAME} {embedding_model_entity.model_name}"
            span = get_tracer().start_span(span_name)
            span.set_attribute(GEN_AI_SPAN_KIND, EMBEDDING_SPAN_KIND)
            span.set_attribute(INPUT_MESSAGES, json.dumps(pydantic_to_dict(messages), ensure_ascii=False))
            span.set_attribute(EMBEDDING_MDOEL_NAME, embedding_model_entity.model_name)
            span.set_attribute(INPUT_VALUE, query)

            span.set_attribute(GEN_AI_OPERATION_NAME, EMBEDDING_OPERATION_NAME)

            query_embedding = await func(self, *args, **kwargs)
            span.set_attribute(EMBEDDING_DIMENSION_COUNT, len(query_embedding))
            span.set_status(STATUS_OK)
            span.end()
            return query_embedding
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise

    return wrapper


def reranker_wrapper(func):
    """decorator to capture input & output string of reranker operation."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(self, *args, **kwargs)

        query = kwargs.get("query", "[unknown]")
        vector_result = kwargs.get("vector_result", None)
        top_n = kwargs.get("top_n", None)
        try:
            span_name = f"{RERANKER_OPERATION_NAME} {self.model}"
            span = get_tracer().start_span(span_name)
            span.set_attribute(GEN_AI_SPAN_KIND, RERANKER_SPAN_KIND)
            span.set_attribute(RERANKER_MODEL_NAME, self.model)
            span.set_attribute(GEN_AI_OPERATION_NAME, RERANKER_OPERATION_NAME)

            messages = [{
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "content": query,
                    }
                ],
            }]
            span.set_attribute(INPUT_MESSAGES, json.dumps(pydantic_to_dict(messages), ensure_ascii=False))

            if vector_result and vector_result.nodes:
                input_documents = [
                    {
                        "id": vector_result.ids[i],
                        "content": vector_result.nodes[i].text,
                        "score": vector_result.similarities[i],
                        "metadata": vector_result.nodes[i].metadata,
                    }
                    for i in range(len(vector_result.nodes))
                ]
            else:
                input_documents = []

            input_value = {
                "documents": input_documents,
                "query": query,
                "document_size": len(input_documents),
                "top_k": top_n,
            }
            span.set_attribute(INPUT_VALUE, json.dumps(pydantic_to_dict(input_value), ensure_ascii=False))

            rerank_result = await func(self, *args, **kwargs)
            output_documents = []
            if rerank_result and rerank_result.nodes:
                output_documents = [
                    {
                        "id": rerank_result.ids[i],
                        "content": rerank_result.nodes[i].text,
                        "score": rerank_result.similarities[i],
                        "metadata": rerank_result.nodes[i].metadata,
                    }
                    for i in range(len(rerank_result.nodes))
                ]
            output_value = {
                "documents": output_documents
            }
            span.set_attribute(OUTPUT_VALUE, json.dumps(pydantic_to_dict(output_value), ensure_ascii=False))
            span.set_status(STATUS_OK)
            span.end()
            return rerank_result
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise

    return wrapper
