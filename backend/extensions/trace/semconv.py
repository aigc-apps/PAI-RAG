"""OpenInference semantic-convention attribute keys and span-kind values.

We hard-code the string constants instead of importing them from
``openinference-semantic-conventions`` so this module (and the manual spans that
use it) has zero third-party dependency and stays importable even when the
openinference package is absent. The strings are the stable OpenInference wire
keys that Langfuse / Arize Phoenix parse natively:

  https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
"""
from __future__ import annotations


# --- span kind ---------------------------------------------------------------
# Attribute whose value tells the UI how to render the span.
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"


class SpanKind:
    AGENT = "AGENT"
    LLM = "LLM"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"


# --- generic input / output --------------------------------------------------
INPUT_VALUE = "input.value"
INPUT_MIME_TYPE = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME_TYPE = "output.mime_type"

MIME_TEXT = "text/plain"
MIME_JSON = "application/json"


# --- LLM ---------------------------------------------------------------------
LLM_MODEL_NAME = "llm.model_name"
LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"


# --- tool --------------------------------------------------------------------
TOOL_NAME = "tool.name"
TOOL_DESCRIPTION = "tool.description"
TOOL_PARAMETERS = "tool.parameters"


# --- session / user / metadata (Langfuse maps these to its trace fields) -----
SESSION_ID = "session.id"
USER_ID = "user.id"
METADATA = "metadata"
TAG_TAGS = "tag.tags"
