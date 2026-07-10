"""Environment-driven tracing configuration.

Pure stdlib (os / base64) — no opentelemetry / third-party imports — so it can be
read and unit-tested even in a lean deployment where the OTel stack is absent.

Two ways to configure the exporter, both via environment variables:

1. Standard OTLP (takes precedence):
   - ``OTEL_EXPORTER_OTLP_ENDPOINT``   e.g. ``http://otel-collector:4318``
   - ``OTEL_EXPORTER_OTLP_HEADERS``    auth, e.g. ``Authorization=Basic <b64>,x-k=v``
   - ``OTEL_EXPORTER_OTLP_PROTOCOL``   ``http/protobuf`` (default) | ``grpc``
   - ``OTEL_SERVICE_NAME``             default ``pai-agent``
   - ``OTEL_TRACES_SAMPLER_ARG``       sampling ratio 0..1, default 1.0
   - ``OTEL_TRACES_ENABLED``           master switch: ``true``/``false``/``auto`` (default
                                       ``auto`` = enabled iff an endpoint resolves)

2. Langfuse convenience (used only when no explicit OTLP endpoint is set):
   - ``LANGFUSE_HOST``                 default ``https://cloud.langfuse.com``
   - ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``
     → endpoint ``{HOST}/api/public/otel`` + ``Authorization: Basic base64(pk:sk)``
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def _parse_headers(raw: str) -> Dict[str, str]:
    """Parse the OTel ``key=value,key2=value2`` header format into a dict.

    Values may themselves contain ``=`` (e.g. base64 padding), so we split on the
    first ``=`` only. Whitespace around keys/values is stripped; empty entries are
    ignored. This mirrors the OTel spec's OTLP header encoding.
    """
    out: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip()
    return out


def _truthy(val: Optional[str]) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class TraceConfig:
    enabled: bool = False
    endpoint: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    protocol: str = "http/protobuf"   # "http/protobuf" | "grpc"
    service_name: str = "pai-agent"
    sample_ratio: float = 1.0

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None,
                 default_service_name: str = "pai-agent") -> "TraceConfig":
        env = env if env is not None else os.environ

        endpoint = (env.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        headers = _parse_headers(env.get("OTEL_EXPORTER_OTLP_HEADERS") or "")
        protocol = (env.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf").strip().lower()

        # Fall back to Langfuse-style keys only when no explicit OTLP endpoint is
        # given. Deriving the endpoint + Basic-auth header from the public/secret
        # key pair is exactly what the Langfuse OTel docs prescribe.
        if not endpoint:
            pk = (env.get("LANGFUSE_PUBLIC_KEY") or "").strip()
            sk = (env.get("LANGFUSE_SECRET_KEY") or "").strip()
            if pk and sk:
                host = (env.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").strip().rstrip("/")
                endpoint = f"{host}/api/public/otel"
                token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
                headers.setdefault("Authorization", f"Basic {token}")
                # Langfuse's OTLP ingest speaks http/protobuf.
                protocol = "http/protobuf"

        service_name = (env.get("OTEL_SERVICE_NAME") or default_service_name).strip() or default_service_name

        try:
            sample_ratio = float(env.get("OTEL_TRACES_SAMPLER_ARG") or "1.0")
        except (TypeError, ValueError):
            sample_ratio = 1.0
        sample_ratio = min(1.0, max(0.0, sample_ratio))

        # Master switch: auto (default) => on iff we resolved an endpoint.
        switch = (env.get("OTEL_TRACES_ENABLED") or "auto").strip().lower()
        if switch in ("0", "false", "no", "off"):
            enabled = False
        elif switch in ("1", "true", "yes", "on"):
            enabled = bool(endpoint)
        else:  # "auto"
            enabled = bool(endpoint)

        return cls(
            enabled=enabled,
            endpoint=endpoint,
            headers=headers,
            protocol=protocol,
            service_name=service_name,
            sample_ratio=sample_ratio,
        )
