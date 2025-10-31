"""Agent configuration management module.

This module provides centralized configuration management for the multi-agent system,
loading settings from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AgentConfig:
    """Centralized configuration for the multi-agent system."""
    
    # ===== LLM 配置 =====
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    
    # ===== MCP 配置 =====
    MCP_AMAP_URL: str = os.getenv("MCP_AMAP_URL", "")
    MCP_AMAP_HEADERS: str = os.getenv("MCP_AMAP_HEADERS", "")
    
    # ===== 知识库配置 =====
    KB_API_URL: str = os.getenv("KB_API_URL", "")
    KB_AUTHORIZATION: str = os.getenv("KB_AUTHORIZATION", "")
    KB_KNOWLEDGE_ID: str = os.getenv("KB_KNOWLEDGE_ID", "")
    
    # ===== Tavily Search 配置 =====
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    
    # ===== ReAct Executor 配置 =====
    REACT_MAX_STEPS: int = int(os.getenv("REACT_MAX_STEPS", "3"))
    REACT_DEFAULT_LANGUAGE: str = os.getenv("REACT_DEFAULT_LANGUAGE", "中文")
    
    # ===== Agent 角色常量 =====
    class AgentRoles:
        """Agent role names."""
        RESEARCHER = "researcher"
        KB_RETRIEVER = "kb_retriever"
        REPORTER = "reporter"
        MAP_NAVIGATOR = "map_navigator"
        
        @classmethod
        def all(cls) -> list[str]:
            """Return all enabled agent roles."""
            return [
                cls.RESEARCHER,
                cls.KB_RETRIEVER,
                cls.REPORTER,
                cls.MAP_NAVIGATOR,
            ]
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration and return list of missing keys.
        
        Returns:
            List of missing required configuration keys.
        """
        missing = []
        
        # KB_AUTHORIZATION is required if KB_API_URL is set
        if cls.KB_API_URL and not cls.KB_AUTHORIZATION:
            missing.append("KB_AUTHORIZATION")
        
        # TAVILY_API_KEY is required for researcher
        if not cls.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY (optional if not using researcher)")
        
        return missing