from typing import Type

from sqlmodel import SQLModel
from db.models.trace import TraceModelEntity
from config.providers.base_provider import BaseConfigProvider
from extensions.trace.base import init_instrument, TraceConfig


class TraceProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = TraceModelEntity

    def _load_entries(self, entries):
        super()._load_entries(entries)
        # 检查是否有数据
        if not entries:
            return
        entry = entries[0]
        config = TraceConfig(
                    service_name=getattr(entry, 'service_name', None),
                    token=getattr(entry, 'token', None),
                    endpoint=getattr(entry, 'endpoint', None),
                    enabled=getattr(entry, 'enabled', None),
                    user_args=getattr(entry, 'user_args', None)
                )
        if config.is_enabled():
            init_instrument(config=config)

    def add(self, entry):
        super().add(entry)

    def update(self, entry):
        super().update(entry)



trace_provider = TraceProvider()
