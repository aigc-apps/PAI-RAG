import asyncio
from sqlalchemy import Float, and_, exists, or_

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.user_role import PermissionEntity, UserRoleEntity
from common.chat.models import MetadataFilteringCondition, Condition
from db.db_context import with_async_db_session
from db.models.knowledgebase.file import KbFileEntity
from loguru import logger

def _build_metadata_condition_(
    condition: Condition,
) -> tuple[str, list[str]]:
    if (condition.comparison_operator not in ["empty", "not empty"] and
        (condition.value is None or condition.value == "")):
        return None

    match condition.comparison_operator:
        case "contains":
            condition_filter = KbFileEntity.file_metadata[condition.name].like(f"%{condition.value}%")
        case "not contains":
            condition_filter = ~KbFileEntity.file_metadata[condition.name].like(f"%{condition.value}%")
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
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().is_(None)
        case "not empty":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().isnot(None)
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
    user_id: str = None,
) -> list[str]:

    # 为了简化实现复杂度，把metadata设定在file这一层
    # TODO: possible limitations: IN clause长度过长导致执行速度慢/超出限制？
    filters = []
    if metadata_filter is not None and metadata_filter.conditions is not None:
        for condition in metadata_filter.conditions:
            condition_filter = _build_metadata_condition_(condition)
            if condition_filter is not None:
                filters.append(condition_filter)

    sub_clauses = [KbFileEntity.active, KbFileEntity.kb_id == kb_id]
    if len(filters) > 0:
        if metadata_filter.logical_operator.lower() == "and":
            sub_clauses.append(and_(*filters))
        else:
            sub_clauses.append(where_clause = or_(*filters))

    # 文档没有指定权限，可公开访问
    has_role_binding = exists().where(PermissionEntity.name == KbFileEntity.id)
    is_document_public = ~has_role_binding
    # 判断用户角色可访问的文档
    is_allowed = exists().where(
        and_(
            PermissionEntity.name == KbFileEntity.id,
            PermissionEntity.role_id == UserRoleEntity.role_id,
            UserRoleEntity.user_id == user_id,
        )
    )
    sub_clauses.append(or_(is_document_public, is_allowed))

    file_entities = (await session.exec(
        select(KbFileEntity).where(and_(*sub_clauses)))).all()

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
