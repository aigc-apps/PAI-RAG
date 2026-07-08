from __future__ import annotations
from typing import Optional

from fastapi import Request

from app.config import Settings, get_settings


def external_base_url(request: Optional[Request] = None,
                      settings: Optional[Settings] = None) -> str:
    """The deployment's externally-reachable base URL, trailing slash stripped.

    Prefers the configured ``PUBLIC_BASE_URL`` so links we hand out (the ROS
    one-click template URL, invite links) point at the canonical public host
    rather than whatever Host/proxy header the current request happened to arrive
    on (often the frontend origin). Falls back to the request's own ``base_url``
    when unset, and to ``""`` when neither is available."""
    settings = settings or get_settings()
    configured = (settings.public_base_url or "").strip().rstrip("/")
    if configured:
        return configured
    if request is not None:
        return str(request.base_url).rstrip("/")
    return ""
