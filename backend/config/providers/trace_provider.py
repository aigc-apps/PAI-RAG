from typing import Type

from sqlmodel import SQLModel
from db.models.trace import TraceModelEntity
from config.providers.base_provider import BaseConfigProvider
from extensions.trace.base import init_instrument


class TraceProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = TraceModelEntity

    def _load_entries(self, entries):
        super()._load_entries(entries)
        # 检查是否有数据
        if not entries:
            return
        entry = entries[0]
        init_instrument(config=entry)

    def add(self, entry):
        super().add(entry)
        init_instrument(config=entry)

    def update(self, entry):
        super().update(entry)
        init_instrument(config=entry)



trace_provider = TraceProvider()
