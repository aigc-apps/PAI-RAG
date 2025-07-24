from loguru import logger


dependency_initialized = False


async def init_dependencies(init_mcp_tools=False):
    global dependency_initialized
    if dependency_initialized:
        return

    from pairag.db.db_context import init_db

    await init_db()
    logger.info("Initialized databases for MCP.")

    if init_mcp_tools:
        from pairag.mcp.providers.mcp_tool_provider import mcp_provider
        from pairag.mcp.providers.websearch_provider import websearch_provider

        await mcp_provider.refresh()
        logger.info("Initialized mcp tools.")
        await websearch_provider.refresh()
        logger.info("Initialized websearch configs.")

    from pairag.mcp.providers.llm_provider import llm_provider
    from pairag.mcp.providers.embedding_provider import embedding_provider
    from pairag.mcp.providers.reranker_provider import reranker_provider
    from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider

    await llm_provider.refresh()
    logger.info("Initialized llm models.")
    await embedding_provider.refresh()
    logger.info("Initialized embedding models.")
    await reranker_provider.refresh()
    logger.info("Initialized reranker models.")
    await knowledgebase_provider.refresh()
    logger.info("Initialized knowledgebases.")

    dependency_initialized = True
