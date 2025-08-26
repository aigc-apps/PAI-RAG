from typing import Any, Type, cast

from extensions.guardrail.guardrail_check import GuardrailChecker
from pydantic import Field
from sqlmodel import SQLModel
from db.encrypt_utils import decrypt_key
from db.models.guardrail import GuardrailConfigEntity
from config.providers.base_provider import BaseConfigProvider


class GuardrailProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = GuardrailConfigEntity
    checker: Any = Field(default=None)

    def _load(self, entry: GuardrailConfigEntity):
        self.checker = GuardrailChecker(
            access_key_id=decrypt_key(entry.encrypted_access_key_id),
            access_key_secret=decrypt_key(entry.encrypted_access_key_secret),
            region_id=entry.region_id,
            endpoint=entry.endpoint,
        )

    def _load_entries(self, entries):
        super()._load_entries(entries)
        if len(entries) > 0:
            self._load(entries[0])

    def add(self, entry):
        super().add(entry)
        self._load(entry)

    def update(self, entry):
        super().update(entry)
        self._load(entry)

    def get_checker(self) -> GuardrailChecker:
        return cast(GuardrailChecker, self.checker)

guardrail_provider = GuardrailProvider()
