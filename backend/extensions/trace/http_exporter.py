# Copyright The OpenTelemetry Authors
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

"""OTLP Span Exporter"""

from os import environ
from typing import Dict, Optional
from urllib.parse import urlparse
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
    OTEL_EXPORTER_OTLP_TRACES_HEADERS,
)
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_HEADERS,
)
from opentelemetry.util.re import parse_env_headers


# pylint: disable=no-member
class ReloadableHttpOTLPSpanExporter(OTLPSpanExporter):
    # pylint: disable=unsubscriptable-object
    """OTLP span exporter

    Args:
        endpoint: OpenTelemetry Collector receiver endpoint
        insecure: Connection type
        credentials: Credentials object for server authentication
        headers: Headers to send when exporting
        timeout: Backend request timeout in seconds
        compression: compression method to use
    """
    def __init__(
        self,
        endpoint: Optional[str] = None,
        headers: Dict[str, str] = None,
        timeout: Optional[int] = None,
    ):
        endpoint = endpoint or environ.get(OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)

        assert endpoint, "endpoint is required"
        endpoint = endpoint.rstrip('/')

        if not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint}/v1/traces"

        headers = headers or environ.get(OTEL_EXPORTER_OTLP_TRACES_HEADERS)
        if isinstance(headers, str):
            headers = parse_env_headers(headers, liberal=True)

        super().__init__(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout,
            compression=Compression.Gzip,
        )

    def reload(
        self,
        endpoint: Optional[str] = None,
        headers: Dict[str, str] | str = None,
    ):
        self._endpoint = endpoint

        parsed_url = urlparse(self._endpoint)

        if parsed_url.netloc:
            self._endpoint = parsed_url.netloc

        self._headers = headers or environ.get(OTEL_EXPORTER_OTLP_HEADERS)
        if isinstance(self._headers, str):
            self._headers = parse_env_headers(self._headers, liberal=True)

        self._session.headers.update(self._headers)
