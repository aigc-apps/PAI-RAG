"""Adapter registry: maps a source_type to its adapter implementation.

New source type = new adapter registered here; the sync worker and the rest of
the pipeline stay unchanged.
"""

from typing import Dict, Optional, Type

from common.knowledgebase.types import DataSourceType
from rag.datasource.base_adapter import BaseAdapter
from rag.datasource.adapters.llms_txt import LlmsTxtAdapter
from rag.datasource.adapters.sphinx import SphinxAdapter

_ADAPTERS: Dict[str, Type[BaseAdapter]] = {
    DataSourceType.llms_txt.value: LlmsTxtAdapter,
    DataSourceType.sphinx.value: SphinxAdapter,
    # DataSourceType.local.value: LocalAdapter,     # later
    # DataSourceType.github.value: GithubAdapter,   # later
}


def get_adapter(
    source_type: str,
    datasource_key: str,
    source_config: Optional[dict] = None,
) -> BaseAdapter:
    """Instantiate the adapter for a source type."""
    key = source_type.value if hasattr(source_type, "value") else str(source_type)
    adapter_cls = _ADAPTERS.get(key)
    if adapter_cls is None:
        raise ValueError(f"No adapter registered for source_type '{key}'.")
    return adapter_cls(datasource_key=datasource_key, source_config=source_config)


def supported_source_types() -> list:
    return list(_ADAPTERS.keys())
