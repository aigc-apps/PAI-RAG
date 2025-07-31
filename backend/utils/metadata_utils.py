from typing import List
from db.models.knowledgebase.knowledgebase import MetadataConfig


def ensure_metadata_configs_is_valid(metadata_configs: List[MetadataConfig]):
    if not metadata_configs:
        return

    seen_metadata_keys = set()
    for metadata_config in metadata_configs:
        if metadata_config.name in seen_metadata_keys:
            raise ValueError(f"Duplicate metadata name: {metadata_config.name}")
        seen_metadata_keys.add(metadata_config.name)
