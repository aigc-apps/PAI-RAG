import os
from urllib.parse import quote_plus, urlunparse
from typing import Optional, List, Tuple
from loguru import logger
from dataclasses import dataclass


@dataclass
class RedisConfig:
    """Redis configuration container supporting both standalone and cluster modes."""

    # Common settings
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    username: str = ""
    db: int = 0
    ssl: bool = False

    # Cluster mode settings
    cluster_mode: bool = False
    cluster_nodes: List[Tuple[str, int]] = None  # [(host1, port1), (host2, port2), ...]

    def __post_init__(self):
        if self.cluster_nodes is None:
            self.cluster_nodes = []


def _parse_cluster_nodes(nodes_str: str) -> List[Tuple[str, int]]:
    """
    Parse cluster nodes from environment variable.

    Supported formats:
    - "host1:port1,host2:port2,host3:port3"
    - "host1:port1;host2:port2;host3:port3"

    Returns:
        List of (host, port) tuples
    """
    if not nodes_str:
        return []

    nodes = []
    # Support both comma and semicolon as delimiters
    delimiter = ";" if ";" in nodes_str else ","
    for node in nodes_str.split(delimiter):
        node = node.strip()
        if not node:
            continue
        if ":" in node:
            host, port_str = node.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                logger.warning(f"Invalid port in cluster node: {node}, skipping")
                continue
            nodes.append((host, port))
        else:
            # Default Redis cluster port
            nodes.append((node, 6379))
    return nodes


def _load_redis_config() -> RedisConfig:
    """Load Redis configuration from environment variables."""
    # Check cluster mode
    cluster_mode = os.getenv("REDIS_CLUSTER_MODE", "false").lower() == "true"

    # Common settings
    host = os.getenv("REDIS_HOST", "localhost") or "localhost"
    port_str = os.getenv("REDIS_PORT", "6379")
    port = int(port_str) if port_str else 6379
    password = os.getenv("REDIS_PASSWORD", "")
    db_str = os.getenv("REDIS_DB")
    db = int(db_str) if db_str else 0
    ssl = os.getenv("REDIS_SSL", "false").lower() == "true"

    # Cluster nodes (only used in cluster mode)
    cluster_nodes_str = os.getenv("REDIS_CLUSTER_NODES", "")
    cluster_nodes = _parse_cluster_nodes(cluster_nodes_str)

    # If cluster mode but no nodes specified, use default host:port as the seed node
    if cluster_mode and not cluster_nodes:
        cluster_nodes = [(host, port)]

    return RedisConfig(
        host=host,
        port=port,
        password=password,
        db=db,
        ssl=ssl,
        cluster_mode=cluster_mode,
        cluster_nodes=cluster_nodes,
    )


def compose_redis_url_safe(
    host: str = "localhost",
    port: int = 6379,
    username: Optional[str] = None,
    password: Optional[str] = None,
    db: int = 0,
    ssl: bool = False,
) -> str:
    """
    Compose standalone Redis URL with special character handling.

    Returns URL format: redis[s]://[username:password@]host:port/db
    """
    scheme = "rediss" if ssl else "redis"

    # Handle authentication (URL encode special characters)
    if username and password:
        netloc = f"{quote_plus(username)}:{quote_plus(password)}@{host}:{port}"
    elif password:
        netloc = f":{quote_plus(password)}@{host}:{port}"
    else:
        netloc = f"{host}:{port}"

    path = f"/{db}"
    url = urlunparse((scheme, netloc, path, "", "", ""))
    return url


def compose_cluster_url(
    nodes: List[Tuple[str, int]],
    username: Optional[str] = None,
    password: Optional[str] = None,
    ssl: bool = False,
) -> str:
    """
    Compose Redis Cluster URL for Celery.

    Celery/kombu requires a single host:port in the URL.
    Additional nodes should be passed via broker_transport_options.
    Note: Redis Cluster does not support DB selection (always uses DB 0)

    Returns URL format: redis[s]+cluster://[username:password@]host:port
    """
    scheme = "rediss+cluster" if ssl else "redis+cluster"

    # Use only the first node in the URL (kombu can't parse multiple hosts in URL)
    if nodes:
        host, port = nodes[0]
    else:
        host, port = "localhost", 6379

    # Handle authentication
    if username and password:
        auth = f"{quote_plus(username)}:{quote_plus(password)}@"
    elif password:
        auth = f":{quote_plus(password)}@"
    else:
        auth = ""

    return f"{scheme}://{auth}{host}:{port}"


def get_redis_url(config: RedisConfig) -> str:
    """Get appropriate Redis URL based on configuration."""
    if config.cluster_mode:
        return compose_cluster_url(
            nodes=config.cluster_nodes,
            username=config.username,
            password=config.password,
            ssl=config.ssl,
        )
    else:
        return compose_redis_url_safe(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
            db=config.db,
            ssl=config.ssl,
        )


# Load configuration
REDIS_CONFIG = _load_redis_config()

# Legacy exports for backward compatibility
REDIS_HOST = REDIS_CONFIG.host
REDIS_PORT = REDIS_CONFIG.port
REDIS_PASSWORD = REDIS_CONFIG.password
REDIS_DB = REDIS_CONFIG.db
REDIS_SSL = REDIS_CONFIG.ssl
REDIS_CLUSTER_MODE = REDIS_CONFIG.cluster_mode
REDIS_CLUSTER_NODES = REDIS_CONFIG.cluster_nodes

# Compose URL based on mode
REDIS_URL = get_redis_url(REDIS_CONFIG)

if REDIS_CLUSTER_MODE:
    logger.info(f"Redis Cluster Mode enabled with nodes: {REDIS_CLUSTER_NODES}")
else:
    logger.info(f"Redis Standalone Mode: {REDIS_HOST}:{REDIS_PORT}")
