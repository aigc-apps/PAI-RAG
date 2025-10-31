"""LLM factory module for creating and caching LLM instances."""
from functools import lru_cache
from langchain_openai import ChatOpenAI
from agent.config import AgentConfig


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Get a singleton LLM instance.
    
    Returns:
        Cached ChatOpenAI instance configured with AgentConfig settings.
    """
    return ChatOpenAI(
        model=AgentConfig.LLM_MODEL,
        temperature=AgentConfig.LLM_TEMPERATURE
    )


def clear_llm_cache():
    """Clear the LLM cache (useful for testing or reconfiguration)."""
    get_llm.cache_clear()
