import dataclasses
import json
import os
import logging
import traceback
from contextlib import asynccontextmanager

from opentelemetry import context as context_api
from opentelemetry.instrumentation.llamaindex.config import Config
from opentelemetry.semconv_ai import SpanAttributes
from pydantic import BaseModel


def _with_tracer_wrapper(func):
    def _with_tracer(tracer):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@asynccontextmanager
async def start_as_current_span_async(tracer, *args, **kwargs):
    with tracer.start_as_current_span(*args, **kwargs) as span:
        yield span


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


def ensure_serializable(data: BaseModel) -> BaseModel:
    """
    Workaround for https://github.com/pydantic/pydantic/issues/7713, see https://github.com/pydantic/pydantic/issues/7713#issuecomment-2604574418
    """
    try:
        json.dumps(data)
    except TypeError:
        # use `vars` to coerce nested data into dictionaries
        data_json_from_dicts = json.dumps(data, default=lambda x: vars(x))  # type: ignore
        data_obj = json.loads(data_json_from_dicts)
        data = type(data)(**data_obj)
    return data


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, BaseModel):
            return ensure_serializable(o).model_dump()
        elif hasattr(o, "dict"):
            return o.dict()
        elif hasattr(o, "json"):
            return o.json()
        elif hasattr(o, "to_json"):
            return o.to_json()
        
        return super().default(o)


@dont_throw
def process_request(span, args, kwargs):
    if should_send_prompts():
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps({"args": args, "kwargs": kwargs}, cls=JSONEncoder),
        )


@dont_throw
def process_response(span, res):
    if should_send_prompts():
        print("++++output1: ", json.dumps(res, cls=JSONEncoder))
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps(res, cls=JSONEncoder),
        )
