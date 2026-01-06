# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Set

from opentelemetry.baggage import get_all as get_all_baggage
from opentelemetry.context import Context
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span


DEFAULT_ALLOWED_PREFIXES = set(["traffic.llm_sdk."])
DEFAULT_STRIP_PREFIXES = set(["traffic.llm_sdk."])


class LoongSuiteBaggageSpanProcessor(SpanProcessor):
    """
    LoongSuite Baggage Span Processor

    Reads Baggage entries from the parent context and adds matching baggage
    key-value pairs to span attributes based on configured prefix matching rules.

    Supported features:
    1. Prefix matching: Only process baggage keys that match specified prefixes
    2. Prefix stripping: Remove specified prefixes before writing to attributes

    Example:
        # Configure matching prefixes: "traffic.", "app."
        # Configure stripping prefix: "traffic."
        # baggage: traffic.hello_key = "value"
        # Result: attributes will have hello_key = "value" (prefix stripped)

        # baggage: app.user_id = "123"
        # Result: attributes will have app.user_id = "123" (app. prefix not stripped)

    ⚠ Warning ⚠️

    Do not put sensitive information in Baggage.

    To repeat: a consequence of adding data to Baggage is that the keys and
    values will appear in all outgoing HTTP headers from the application.
    """

    def __init__(
        self,
        allowed_prefixes: Optional[Set[str]] = DEFAULT_ALLOWED_PREFIXES,
        strip_prefixes: Optional[Set[str]] = DEFAULT_STRIP_PREFIXES,
    ) -> None:
        """
        Initialize LoongSuite Baggage Span Processor

        Args:
            allowed_prefixes: Set of allowed baggage key prefixes. If None or empty,
                             all baggage keys are allowed. If specified, only keys
                             matching these prefixes will be processed.
            strip_prefixes: Set of prefixes to strip. If a baggage key matches these
                           prefixes, they will be removed before writing to attributes.
        """
        self._allowed_prefixes = allowed_prefixes or []
        self._strip_prefixes = strip_prefixes or set()

        # If allowed_prefixes is empty, allow all prefixes
        self._allow_all = len(self._allowed_prefixes) == 0

    def _should_process_key(self, key: str) -> bool:
        """
        Determine whether this baggage key should be processed

        Args:
            key: baggage key

        Returns:
            True if the key should be processed, False otherwise
        """
        if self._allow_all:
            return True

        # Check if key matches any of the allowed prefixes
        for prefix in self._allowed_prefixes:
            if key.startswith(prefix):
                return True

        return False

    def _strip_prefix(self, key: str) -> str:
        """
        Strip matching prefix from key

        Args:
            key: original baggage key

        Returns:
            key with prefix stripped
        """
        for prefix in self._strip_prefixes:
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    def on_start(
        self, span: "Span", parent_context: Optional[Context] = None
    ) -> None:
        """
        Called when a span starts, adds matching baggage entries to span attributes

        Args:
            span: span to add attributes to
            parent_context: parent context used to retrieve baggage
        """
        baggage = get_all_baggage(parent_context)

        for key, value in baggage.items():
            # Check if this key should be processed
            if not self._should_process_key(key):
                continue

            # Strip prefix if needed
            attribute_key = self._strip_prefix(key)

            # Add to span attributes
            # Baggage values are strings, which are valid AttributeValue
            span.set_attribute(attribute_key, value)  # type: ignore[arg-type]
