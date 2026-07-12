from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agent_config import (
    DEFAULT_DOCUMENT,
    AgentConfigDocument,
    _merge_default,
    authored_config_dict,
)
from app.models import AppConfigDocumentRow, AppConfigRevisionRow


DEFAULT_CONFIG_ID = "default"
CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredAgentConfig:
    doc: AgentConfigDocument
    revision: int
    checksum: str


@dataclass(frozen=True)
class AgentConfigRevision:
    revision: int
    document: Dict[str, Any]
    checksum: str
    updated_by: Optional[str]


def _checksum(document: Dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SqlAgentConfigStore:
    """SQL-backed source of truth for the authored agent configuration."""

    def __init__(
        self,
        engine,
        *,
        config_id: str = DEFAULT_CONFIG_ID,
        seed: Optional[AgentConfigDocument] = None,
    ):
        self._engine = engine
        self._config_id = config_id
        self._seed = seed or DEFAULT_DOCUMENT.model_copy(deep=True)

    async def load(self) -> StoredAgentConfig:
        async with AsyncSession(self._engine) as session:
            row = await session.get(AppConfigDocumentRow, self._config_id)
            if row is None:
                row = self._bootstrap_row()
                session.add(row)
                session.add(self._revision_row(row))
                await session.commit()
                await session.refresh(row)
        return StoredAgentConfig(
            doc=_merge_default(row.document_json or {}),
            revision=row.revision,
            checksum=row.checksum,
        )

    async def save(
        self,
        doc: AgentConfigDocument,
        *,
        updated_by: Optional[str] = None,
    ) -> StoredAgentConfig:
        document = authored_config_dict(doc)
        checksum = _checksum(document)
        async with AsyncSession(self._engine) as session:
            row = await session.get(AppConfigDocumentRow, self._config_id)
            if row is None:
                row = self._bootstrap_row()
                session.add(row)
                session.add(self._revision_row(row))
                await session.flush()
            row.document_json = document
            row.revision += 1
            row.checksum = checksum
            row.updated_by = updated_by
            session.add(row)
            session.add(self._revision_row(row))
            await session.commit()
            await session.refresh(row)
        return StoredAgentConfig(
            doc=_merge_default(row.document_json or {}),
            revision=row.revision,
            checksum=row.checksum,
        )

    async def list_revisions(self) -> List[AgentConfigRevision]:
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.exec(
                    select(AppConfigRevisionRow)
                    .where(AppConfigRevisionRow.config_id == self._config_id)
                    .order_by(AppConfigRevisionRow.revision)
                )
            ).all()
        return [
            AgentConfigRevision(
                revision=row.revision,
                document=row.document_json or {},
                checksum=row.checksum,
                updated_by=row.updated_by,
            )
            for row in rows
        ]

    async def current_revision(self) -> int:
        async with AsyncSession(self._engine) as session:
            row = await session.get(AppConfigDocumentRow, self._config_id)
            if row is None:
                return 0
            return row.revision

    def _bootstrap_row(self) -> AppConfigDocumentRow:
        document = authored_config_dict(self._seed)
        return AppConfigDocumentRow(
            id=self._config_id,
            schema_version=CONFIG_SCHEMA_VERSION,
            document_json=document,
            revision=1,
            checksum=_checksum(document),
        )

    def _revision_row(self, row: AppConfigDocumentRow) -> AppConfigRevisionRow:
        return AppConfigRevisionRow(
            config_id=row.id,
            revision=row.revision,
            schema_version=row.schema_version,
            document_json=row.document_json or {},
            checksum=row.checksum,
            updated_by=row.updated_by,
            updated_at=row.updated_at,
        )
