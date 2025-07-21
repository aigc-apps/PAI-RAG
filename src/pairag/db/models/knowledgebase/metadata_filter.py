import asyncio
from collections.abc import Sequence
from typing import Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Float, and_, or_

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.db_context import with_async_db_session
from pairag.db.models.knowledgebase.file import KbFileEntity
from loguru import logger

SupportedComparisonOperator = Literal[
    # for string or array
    "contains",
    "not contains",
    "start with",
    "end with",
    "is",
    "is not",
    "empty",
    "not empty",
    # for number
    "=",
    "≠",
    ">",
    "<",
    "≥",
    "≤",
    # for time
    "before",
    "after",
]


class Condition(BaseModel):
    """
    Condition detail
    """

    name: str
    comparison_operator: SupportedComparisonOperator
    value: str | Sequence[str] | None | int | float = None


class MetadataFilteringCondition(BaseModel):
    """
    Metadata Filtering Condition.
    """

    logical_operator: Optional[Literal["and", "or"]] = "and"
    conditions: Optional[list[Condition]] = Field(default=None, deprecated=True)


def _build_metadata_condition_(
    condition: Condition,
) -> tuple[str, list[str]]:

    if condition.value is None or condition.value == "":
        return None

    match condition.comparison_operator:
        case "contains":
            condition_filter = KbFileEntity.file_metadata[condition.name].like(f"%{condition.value}%")
        case "not contains":
            condition_filter = KbFileEntity.file_metadata[condition.name].like(f"%{condition.value}%")
        case "start with":
            condition_filter = KbFileEntity.file_metadata[condition.name].like(f'"{condition.value}%')
        case "end with":
            condition_filter = KbFileEntity.file_metadata[condition.name].like(f'%{condition.value}"')
        case "is" | "=":
            if isinstance(condition.value, str):
                # 添加json_quote ""
                condition_filter = KbFileEntity.file_metadata[condition.name] == f'"{condition.value}"'
            else:
                condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Float) == condition.value
        case "is not" | "≠":
            if isinstance(condition.value, str):
                # 添加json_quote ""
                condition_filter = KbFileEntity.file_metadata[condition.name] != f'"{condition.value}"'
            else:
                condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Float) != condition.value
        case "empty":
            condition_filter = KbFileEntity.file_metadata[condition.name].is_(None)
        case "not empty":
            condition_filter = KbFileEntity.file_metadata[condition.name].isnot(None)
        case "before" | "<":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Float) < condition.value
        case "after" | ">":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Float) > condition.value
        case "≤" | "<=":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Float) <= condition.value
        case "≥" | ">=":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Float) >= condition.value
        case _:
            logger.warning(f"Unknown operator {condition.comparison_operator}")
            return None

    return condition_filter

@with_async_db_session
async def query_file_ids_with_metadata_filter(
    session: AsyncSession,
    kb_id: str,
    metadata_filter: MetadataFilteringCondition,
) -> list[str]:
    if metadata_filter is None or metadata_filter.conditions is None:
        return []

    # 为了简化实现复杂度，把metadata设定在file这一层
    # TODO: possible limitations: IN clause长度过长导致执行速度慢/超出限制？
    filters = [and_(KbFileEntity.active, KbFileEntity.kb_id == kb_id)]

    for condition in metadata_filter.conditions:
        condition_filter = _build_metadata_condition_(condition)
        if condition_filter is not None:
            filters.append(condition_filter)

    if metadata_filter.logical_operator.lower() == "and":
        where_clause = and_(*filters)
    else:
        where_clause = or_(*filters)
    file_entities = (await session.exec(
        select(KbFileEntity)
        .where(where_clause))).all()
    file_ids = [entity.id for entity in file_entities]
    return file_ids


if __name__ == '__main__':
    metadata_filters = MetadataFilteringCondition(
        logical_operator="and",
        conditions=[
            Condition(
                name="file_name",
                value="花",
                comparison_operator="start with",
            )
        ]
    )
    file_ids = asyncio.run(query_file_ids_with_metadata_filter("b84035e1ca244797ae1830d06b9780d7", metadata_filters))
    print("file_ids", file_ids)
