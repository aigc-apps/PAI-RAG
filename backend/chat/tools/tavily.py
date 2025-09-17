from dataclasses import dataclass

# https://docs.tavily.com/documentation/api-reference/endpoint/search#response-results-favicon

try:
    from tavily import AsyncTavilyClient
except ImportError as _import_error:
    raise ImportError(
        'Please install `tavily-python` to use the Tavily search tool, '
        'you can use the `tavily` optional group — `pip install "pydantic-ai-slim[tavily]"`'
    ) from _import_error


DEFAULT_MAX_SEARCH_RESULT = 10


@dataclass
class TavilySearchTool:
    """The Tavily search tool."""

    client: AsyncTavilyClient
    """The Tavily search client."""
    def __init__(
        self,
        api_key: str,
        search_count: int = DEFAULT_MAX_SEARCH_RESULT,
    ):
        self.client = AsyncTavilyClient(api_key)
        self.search_count = search_count


    async def aquery(
        self,
        query: str,
    ):
        """Searches Tavily for the given query and returns the results.

        Args:
            query: The search query to execute with Tavily.

        Returns:
            The search results.
        """
        results = await self.client.search(query, max_results=self.search_count, search_depth='basic', topic='general', time_range=None, include_favicon=True)  # type: ignore[reportUnknownMemberType]
        return {"result": results['results']}  # type: ignore[reportUnknownMemberType]
