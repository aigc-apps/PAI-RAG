# Redis Configuration Guide

PAI-RAG supports both **Redis Standalone** and **Redis Cluster** modes for message broker (Celery) and caching. This document describes how to configure Redis connections via environment variables.

## Environment Variables

### Common Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_HOST` | Redis server hostname or IP address | `localhost` | No |
| `REDIS_PORT` | Redis server port | `6379` | No |
| `REDIS_PASSWORD` | Redis authentication password | (empty) | No |
| `REDIS_DB` | Redis database number (0-15, standalone mode only) | `0` | No |
| `REDIS_SSL` | Enable SSL/TLS connection | `false` | No |

### Cluster Mode Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_CLUSTER_MODE` | Enable Redis Cluster mode | `false` | No |
| `REDIS_CLUSTER_NODES` | Comma-separated list of cluster nodes | (empty) | No |

### Optional Override

| Variable | Description | Default |
|----------|-------------|---------|
| `PAIRAG_BROKER` | Override the auto-generated Redis URL for Celery broker | (empty) |

---

## Configuration Examples

### 1. Local Redis Standalone (Development)

For local development with default Redis installation:

```bash
# .env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

Or simply leave all Redis settings empty to use defaults.

### 2. Redis Standalone with Authentication

For a standalone Redis server with password authentication:

```bash
# .env
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password
REDIS_DB=0
```

**Note:** Special characters in passwords are automatically URL-encoded.

### 3. Redis Standalone with SSL

For secure connections (e.g., cloud-hosted Redis):

```bash
# .env
REDIS_HOST=redis.example.com
REDIS_PORT=6380
REDIS_PASSWORD=your_secure_password
REDIS_SSL=true
```

### 4. Local Redis Cluster

For a local Redis Cluster setup (e.g., development/testing):

```bash
# .env
REDIS_HOST=127.0.0.1
REDIS_PORT=7001
REDIS_PASSWORD=
REDIS_CLUSTER_MODE=true
REDIS_CLUSTER_NODES=127.0.0.1:7001,127.0.0.1:7002,127.0.0.1:7003,127.0.0.1:7004,127.0.0.1:7005,127.0.0.1:7006
```

### 5. Alibaba Cloud Redis Cluster

For Alibaba Cloud Redis (ApsaraDB for Redis) cluster instance:

```bash
# .env
REDIS_HOST=r-xxxxxx.redis.rds.aliyuncs.com
REDIS_PORT=6379
REDIS_PASSWORD=your_password
REDIS_CLUSTER_MODE=true
# No need to specify REDIS_CLUSTER_NODES for cloud Redis - it handles routing internally
```

### 6. AWS ElastiCache Redis Cluster

For AWS ElastiCache Redis cluster mode:

```bash
# .env
REDIS_HOST=your-cluster.xxxxxx.clustercfg.region.cache.amazonaws.com
REDIS_PORT=6379
REDIS_PASSWORD=your_auth_token
REDIS_SSL=true
REDIS_CLUSTER_MODE=true
```

---

## Cluster Nodes Format

The `REDIS_CLUSTER_NODES` variable accepts nodes in the following formats:

**Comma-separated (recommended):**
```
REDIS_CLUSTER_NODES=host1:port1,host2:port2,host3:port3
```

**Semicolon-separated:**
```
REDIS_CLUSTER_NODES=host1:port1;host2:port2;host3:port3
```

**Port defaults to 6379 if not specified:**
```
REDIS_CLUSTER_NODES=host1,host2,host3
```

---

## Important Notes

### Standalone Mode
- Supports database selection via `REDIS_DB` (0-15)
- Generated URL format: `redis[s]://[user:password@]host:port/db`

### Cluster Mode
- **Database selection is NOT supported** - Redis Cluster always uses DB 0
- `REDIS_DB` setting is ignored in cluster mode
- Generated URL format: `redis[s]+cluster://[user:password@]host:port`
- If `REDIS_CLUSTER_NODES` is empty, uses `REDIS_HOST:REDIS_PORT` as the seed node

### Cloud Redis Services
- Most cloud Redis services (Alibaba Cloud, AWS ElastiCache, Azure Cache) handle cluster routing internally
- For cloud services, you typically only need to set `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, and `REDIS_CLUSTER_MODE=true`
- No need to specify individual cluster nodes - the service endpoint handles routing

### Celery Worker Considerations
- In cluster mode, the Celery result backend is disabled due to compatibility issues
- Tasks are treated as fire-and-forget (results are not stored)
- This is suitable for PAI-RAG's file processing tasks which don't require result retrieval

---

## Troubleshooting

### MovedError in Cluster Mode
If you see `redis.exceptions.MovedError`, ensure `REDIS_CLUSTER_MODE=true` is set.

### Port Parsing Error
If you see "Port could not be cast to integer value", this indicates a URL parsing issue. Ensure:
- `REDIS_CLUSTER_NODES` format is correct
- No extra spaces or invalid characters

### Connection Timeout
- Check network connectivity to Redis server
- Verify firewall rules allow the Redis port
- For SSL connections, ensure certificates are valid

### Authentication Failed
- Verify `REDIS_PASSWORD` is correct
- Special characters in passwords are automatically URL-encoded
