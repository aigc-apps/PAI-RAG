from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from app.agent_config import AgentConfigDocument


def _secret(settings: Dict[str, Any]) -> str:
    direct = str(settings.get("api_key") or "")
    if direct and direct != "********":
        return direct
    env_name = str(settings.get("api_key_env") or "")
    return os.environ.get(env_name, "") if env_name else ""


class TavilySearchProvider:
    def __init__(self, *, api_key: str, endpoint: str, defaults: Dict[str, Any]):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.tavily.com/search"
        self.defaults = defaults

    async def search(self, query: str, num_results: int) -> List[Dict]:
        payload = {
            "query": query,
            "max_results": num_results or int(self.defaults.get("max_results") or 5),
            "search_depth": self.defaults.get("search_depth") or "basic",
            "include_raw_content": bool(self.defaults.get("include_raw_content") or False),
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "") or item.get("snippet", ""),
                "score": item.get("score"),
                "source": "tavily",
                "raw_content": item.get("raw_content"),
            }
            for item in data.get("results", [])
        ]


class BraveSearchProvider:
    def __init__(self, *, api_key: str, endpoint: str, defaults: Dict[str, Any]):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.search.brave.com/res/v1/web/search"
        self.defaults = defaults

    async def search(self, query: str, num_results: int) -> List[Dict]:
        params = {
            "q": query,
            "count": num_results or int(self.defaults.get("max_results") or 5),
            "country": self.defaults.get("country") or "US",
            "safesearch": self.defaults.get("safesearch") or "moderate",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                self.endpoint,
                headers={"X-Subscription-Token": self.api_key},
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "source": "brave",
            }
            for item in ((data.get("web") or {}).get("results") or [])
        ]


def make_search_provider(doc: Optional[AgentConfigDocument]):
    if doc is None:
        return None
    caps = {cap.id: cap for cap in doc.capabilities}
    providers = {provider.id: provider for provider in doc.providers}
    search = caps.get("search")
    provider = providers.get("search.default")
    # ``enabled`` is a deprecated no-op; a capability is off only when explicitly
    # permission="disabled". Provider creation then depends on real settings below.
    if search is None or provider is None or getattr(search, "permission", "") == "disabled":
        return None
    settings = provider.settings
    provider_name = str(settings.get("provider") or "none").lower()
    api_key = _secret(settings)
    if provider_name == "none" or not api_key:
        return None
    if provider_name == "tavily":
        return TavilySearchProvider(
            api_key=api_key,
            endpoint=str(settings.get("endpoint") or ""),
            defaults=settings,
        )
    if provider_name == "brave":
        return BraveSearchProvider(
            api_key=api_key,
            endpoint=str(settings.get("endpoint") or ""),
            defaults=settings,
        )
    return None
