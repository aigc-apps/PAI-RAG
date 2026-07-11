"""Adapter registry: maps a source_type to its adapter implementation.

New source type = new adapter registered here; the sync loop and the rest of the
pipeline stay unchanged. MVP registers only ``llms_txt`` (the Aliyun docs source);
sphinx / generic-website / OSS adapters are future work.
"""

from typing import Dict, Optional, Type

from app.datasource.base_adapter import BaseAdapter
from app.datasource.adapters.llms_txt import LlmsTxtAdapter
from app.datasource.adapters.yuque import YuqueAdapter

_ADAPTERS: Dict[str, Type[BaseAdapter]] = {
    LlmsTxtAdapter.source_type: LlmsTxtAdapter,
    YuqueAdapter.source_type: YuqueAdapter,
    # "sphinx": SphinxAdapter,      # later
    # "website": WebsiteAdapter,    # later
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
