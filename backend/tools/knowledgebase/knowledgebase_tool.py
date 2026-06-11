from service.knowledgebase.rag_service import RagService
from typing import Annotated, List, Optional
from functools import partial
from llama_index.core.tools import FunctionTool
import json
from loguru import logger
from common.chat.models import MetadataFilteringCondition


async def aget_knowledgebase_result(
    query: str,
    kb_id: str,
    user_id: str | None = None,
    rag_service: RagService | None = None,
    tenant_id: str = None,
    base_metadata_condition: Optional[MetadataFilteringCondition] = None,
    metadata_condition: Optional[MetadataFilteringCondition] = None,
) -> str:
    """Search knowledgebase with optional metadata filtering."""
    logger.info(f"Searching knowledgebase with kb {kb_id} and user {user_id}.")
    merged_condition = MetadataFilteringCondition.merge(base_metadata_condition, metadata_condition)
    records = await rag_service.aquery(query=query, kb_id=kb_id, user_id=user_id, tenant_id=tenant_id, metadata_condition=merged_condition)
    records_dict = [record.model_dump() for record in records]
    return json.dumps({"result": records_dict}, ensure_ascii=False)


_OPERATORS_BY_TYPE = {
    "string": ["is", "is not", "contains", "not contains", "start with", "end with", "empty", "not empty"],
    "number": ["=", "≠", ">", "<", "≥", "≤"],
    "list": ["in", "not in"],
    "time": ["before", "after"],
}


def _build_metadata_description(schema: List[dict]) -> str:
    """Build the metadata schema section for tool description."""
    lines = ["\n该知识库支持按以下元数据字段过滤搜索范围（通过 metadata_condition 参数）："]
    for field in schema:
        name = field["name"]
        vtype = field.get("value_type", "string")
        desc = field.get("description", "")
        samples = field.get("sample_values", [])
        operators = _OPERATORS_BY_TYPE.get(vtype, _OPERATORS_BY_TYPE["string"])
        line = f"- {name} ({vtype}), 支持操作符: {', '.join(operators)}"
        if desc:
            line += f": {desc}"
        if samples:
            line += f"。已有值: {', '.join(str(s) for s in samples)}"
        lines.append(line)
    return "\n".join(lines)


async def aget_knowledgebase_tool(
    kb_id: str,
    tenant_id: str,
    user_id: Optional[str] = None,
    rag_service: RagService = None,
    metadata_condition: Optional[MetadataFilteringCondition] = None,
    enable_auto_metadata_filter: bool = False,
):
    aquery_knowledgebase_func = partial(
        aget_knowledgebase_result,
        kb_id=kb_id,
        user_id=user_id,
        rag_service=rag_service,
        tenant_id=tenant_id,
        base_metadata_condition=metadata_condition,
    )
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    base_description = (
        f"Search the private knowledgebase for information relevant to the user's query.\n"
        f"\n"
        f"# Knowledgebase info\n"
        f"- Name: {knowledgebase.name}\n"
        f"- Description: {knowledgebase.description}\n"
        f"\n"
        f"# When to use\n"
        f"You MUST call this tool whenever the user's query could be answered or enriched by the content in this knowledgebase. This includes:\n"
        f"- Domain-specific questions that match the knowledgebase description\n"
        f"- Questions about internal documents, policies, products, or proprietary data\n"
        f"- Any query where the knowledgebase may contain more accurate or detailed information than general knowledge\n"
        f"- Follow-up questions on topics previously answered using this knowledgebase — always re-search to get the most relevant context\n"
        f"\n"
        f"# Parameters\n"
        f"- **query** (required, string): Rewrite the user's question into a clear, standalone search query. Resolve pronouns, add necessary context from the conversation, and make the query self-contained.\n"
        f"\n"
        f"# Returns\n"
        f"- A JSON list of matched document chunks with text content, metadata, and relevance scores.\n"
    )

    # Determine whether to expose metadata_condition to the model
    metadata_schema = None
    if enable_auto_metadata_filter:
        from service.cache.metadata_schema_cache import metadata_schema_cache
        metadata_schema = await metadata_schema_cache.get_schema(tenant_id, kb_id)

    if metadata_schema:
        # Schema available — expose metadata_condition with enriched description
        description = base_description + _build_metadata_description(metadata_schema)

        async def query_knowledgebase_handler(
            query: Annotated[
                str,
                "根据上下文添加必要的背景信息，改写一个新的独立问题，使问题更完整，注意指代消解、完善主语等",
            ] = "",
            metadata_condition: Annotated[
                Optional[MetadataFilteringCondition],
                "可选的元数据过滤条件，用于按文件属性筛选搜索范围。"
                "仅当用户明确要求按特定属性筛选时使用，不要自行猜测过滤条件。"
                '示例: {"logical_operator": "and", "conditions": [{"name": "category", "comparison_operator": "is", "value": "武侠"}]}',
            ] = None,
        ):
            # LLM tool calls may produce raw dicts or JSON strings; convert to Pydantic model
            if metadata_condition is not None and not isinstance(metadata_condition, MetadataFilteringCondition):
                if isinstance(metadata_condition, str):
                    metadata_condition = MetadataFilteringCondition(**json.loads(metadata_condition))
                elif isinstance(metadata_condition, dict):
                    metadata_condition = MetadataFilteringCondition(**metadata_condition)
            return await aquery_knowledgebase_func(
                query=query,
                metadata_condition=metadata_condition,
            )
    else:
        # No schema or switch off — only expose query
        description = base_description

        async def query_knowledgebase_handler(
            query: Annotated[
                str,
                "根据上下文添加必要的背景信息，改写一个新的独立问题，使问题更完整，注意指代消解、完善主语等",
            ] = "",
        ):
            return await aquery_knowledgebase_func(
                query=query,
            )

    search_knowledgebase_tool = FunctionTool.from_defaults(
        async_fn=query_knowledgebase_handler,
        name=f"search-knowledgebase-{kb_id[:10]}",
        description=description,
        return_direct=False,
    )
    return search_knowledgebase_tool
