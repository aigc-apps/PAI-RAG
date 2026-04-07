from sqlalchemy import Double, and_, exists, or_

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.user_role import PermissionEntity, UserRoleEntity
from common.chat.models import MetadataFilteringCondition, Condition
from db.models.knowledgebase.file import KbFileEntity
from loguru import logger


# 当筛选出的符合文件列表为空时，raise EmptyFilesException
# 1. 文件集合为空
# 2. metadata未过滤到
# 3. 用户无权限访问
class EmptyFilesException(Exception):
    pass


def _build_metadata_condition_(
    condition: Condition,
) -> tuple[str, list[str]]:
    if (condition.comparison_operator not in ["empty", "not empty"] and
        (condition.value is None or condition.value == "")):
        return None

    if condition.comparison_operator in ["in", "not in"] and not isinstance(condition.value, (list, tuple)):
        logger.warning(f"Operator '{condition.comparison_operator}' requires a list value, got {type(condition.value)}")
        return None

    match condition.comparison_operator:
        case "contains":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().like(f"%{condition.value}%")
        case "not contains":
            condition_filter = or_(
                KbFileEntity.file_metadata[condition.name].as_string().is_(None),
                ~KbFileEntity.file_metadata[condition.name].as_string().like(f"%{condition.value}%")
            )
        case "start with":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().like(f"{condition.value}%")
        case "end with":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().like(f"%{condition.value}")
        case "is" | "=":
            if isinstance(condition.value, str):
                # 添加json_quote ""
                condition_filter = KbFileEntity.file_metadata[condition.name].as_string() == f'{condition.value}'
            else:
                condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Double) == condition.value
        case "is not" | "≠":
            if isinstance(condition.value, str):
                # 添加json_quote ""
                condition_filter = or_(
                    KbFileEntity.file_metadata[condition.name].as_string().is_(None),
                    KbFileEntity.file_metadata[condition.name].as_string() != f'{condition.value}'
                )
            else:
                condition_filter = or_(
                    KbFileEntity.file_metadata[condition.name].as_string().cast(Double).is_(None),
                    KbFileEntity.file_metadata[condition.name].as_string().cast(Double) != condition.value
                )
        case "empty":
            condition_filter = or_(
                KbFileEntity.file_metadata[condition.name].as_string().is_(None),
                KbFileEntity.file_metadata[condition.name].as_string() == ''
            )
        case "not empty":
            condition_filter = and_(
                KbFileEntity.file_metadata[condition.name].as_string().isnot(None),
                KbFileEntity.file_metadata[condition.name].as_string() != ''
            )
        case "in":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().in_(
                [str(v) for v in condition.value]
            )
        case "not in":
            condition_filter = or_(
                KbFileEntity.file_metadata[condition.name].as_string().is_(None),
                ~KbFileEntity.file_metadata[condition.name].as_string().in_(
                    [str(v) for v in condition.value]
                )
            )
        case "before" | "<":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Double) < condition.value
        case "after" | ">":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Double) > condition.value
        case "≤" | "<=":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Double) <= condition.value
        case "≥" | ">=":
            condition_filter = KbFileEntity.file_metadata[condition.name].as_string().cast(Double) >= condition.value
        case _:
            logger.warning(f"Unknown operator {condition.comparison_operator}")
            return None

    return condition_filter


def _build_metadata_filter_recursive(
    metadata_filter: MetadataFilteringCondition,
    max_depth: int = 5,
    _current_depth: int = 0,
):
    """递归构建嵌套的metadata filter条件"""
    if _current_depth >= max_depth:
        raise ValueError(f"Metadata filter nesting depth exceeds maximum of {max_depth}")

    parts = []

    # 处理叶子条件
    if metadata_filter.conditions:
        for condition in metadata_filter.conditions:
            condition_filter = _build_metadata_condition_(condition)
            if condition_filter is not None:
                parts.append(condition_filter)

    # 递归处理嵌套的condition_groups
    if metadata_filter.condition_groups:
        for group in metadata_filter.condition_groups:
            group_filter = _build_metadata_filter_recursive(group, max_depth, _current_depth + 1)
            if group_filter is not None:
                parts.append(group_filter)

    if not parts:
        return None

    if len(parts) == 1:
        return parts[0]

    if metadata_filter.logical_operator and metadata_filter.logical_operator.lower() == "or":
        return or_(*parts)
    return and_(*parts)


# 返回可以搜索的文档id list
# 当metadata_filter为空时，返回所有文档id，返回空列表
# 当metadata_filter不为空时，返回符合条件的文档id list
# 特别的，当筛选出的符合文件列表为空时，raise EmptyFilesException
async def query_file_ids_with_metadata_filter(
    session: AsyncSession,
    kb_id: str,
    metadata_filter: MetadataFilteringCondition,
    user_id: str = None,
) -> list[str]:
    has_no_filter = metadata_filter is None or (
        not metadata_filter.conditions and not metadata_filter.condition_groups
    )
    if has_no_filter and not user_id:
        return []

    # 为了简化实现复杂度，把metadata设定在file这一层
    # TODO: possible limitations: IN clause长度过长导致执行速度慢/超出限制？
    sub_clauses = [KbFileEntity.active, KbFileEntity.kb_id == kb_id]
    if metadata_filter is not None:
        combined_filter = _build_metadata_filter_recursive(metadata_filter)
        if combined_filter is not None:
            sub_clauses.append(combined_filter)

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
    if not file_ids:
        logger.warning(f"No files found with the given metadata filter. sub_clauses: {sub_clauses}")
        raise EmptyFilesException("No files found with the given metadata filter.")

    return file_ids
