#!/bin/bash

# ===================================================================
# API-only 启动脚本
# 支持参数：
#   --port                API 端口 (默认: 8682)
#   --api-instances       API 实例数量 (默认: 1)
#   --worker-instances    Worker 实例数量 (默认: 2)
#   --no-redis            不启动 Redis (使用外部 Redis)
#   --no-worker           不启动 Celery Worker
#   --help                显示帮助
# ===================================================================

# 默认值
PORT=${PORT:-8680}
API_INSTANCE_COUNT=${API_INSTANCE_COUNT:-1}
WORKER_INSTANCE_COUNT=${WORKER_INSTANCE_COUNT:-2}
START_REDIS=${START_REDIS:-true}
START_WORKER=${START_WORKER:-true}

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
        echo "错误: --port 必须是 1-65535 之间的有效端口号"
        exit 1
      fi
      shift 2
      ;;
    --api-instances)
      API_INSTANCE_COUNT="$2"
      if ! [[ "$API_INSTANCE_COUNT" =~ ^[0-9]+$ ]] || [ "$API_INSTANCE_COUNT" -lt 1 ]; then
        echo "错误: --api-instances 必须是正整数"
        exit 1
      fi
      shift 2
      ;;
    --worker-instances|-w)
      WORKER_INSTANCE_COUNT="$2"
      if ! [[ "$WORKER_INSTANCE_COUNT" =~ ^[0-9]+$ ]] || [ "$WORKER_INSTANCE_COUNT" -lt 0 ]; then
        echo "错误: --worker-instances 必须是非负整数"
        exit 1
      fi
      shift 2
      ;;
    --no-redis)
      START_REDIS=false
      shift 1
      ;;
    --no-worker)
      START_WORKER=false
      shift 1
      ;;
    --help|-h)
      echo "用法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --port PORT            API 服务端口 (默认: 8682)"
      echo "  --api-instances N      启动 N 个 API 实例 (默认: 1)"
      echo "  --worker-instances N   启动 N 个 Worker 实例 (默认: 2)"
      echo "  --no-redis             不启动内置 Redis (使用外部 Redis)"
      echo "  --no-worker            不启动 Celery Worker"
      echo "  --help                 显示此帮助信息"
      echo ""
      echo "环境变量:"
      echo "  PORT                   API 服务端口"
      echo "  API_INSTANCE_COUNT     API 实例数量"
      echo "  WORKER_INSTANCE_COUNT  Worker 实例数量"
      echo "  REDIS_URL              外部 Redis 连接地址"
      echo ""
      echo "示例:"
      echo "  $0 --port 8080 --api-instances 2 --worker-instances 3"
      echo "  $0 --port 8080 --no-redis --no-worker"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      echo "使用 --help 查看帮助"
      exit 1
      ;;
  esac
done

# ===================================================================
# 开始启动服务
# ===================================================================

echo "🚀 API 服务启动配置："
echo "   API 端口: $PORT"
echo "   API 实例数: $API_INSTANCE_COUNT"
echo "   Worker 实例数: $WORKER_INSTANCE_COUNT"
echo "   启动 Redis: $START_REDIS"
echo "   启动 Worker: $START_WORKER"
echo ""
echo "🚀 启动服务中..."

# 启动 API 实例
start_api() {
  echo "👉 启动 API 实例 on port $PORT"
  export BACKEND_PORT=$PORT
  gunicorn -w $API_INSTANCE_COUNT -b "0.0.0.0:${PORT}" -c scripts/gunicorn.conf.py app.main:app --timeout 600 &
  API_PID=$!
}

# 启动 Worker 实例
start_worker() {
  if [[ "$START_WORKER" == true ]] && [[ "$WORKER_INSTANCE_COUNT" -gt 0 ]]; then
    echo "👉 启动 Celery Worker (实例数: $WORKER_INSTANCE_COUNT)"
    celery -A app.worker worker --loglevel=info -c $WORKER_INSTANCE_COUNT &
    WORKER_PID=$!
    echo "Celery 已启动 (PID: $WORKER_PID)"
  else
    echo "⏭️  跳过 Celery Worker"
  fi
}

# 定义清理函数
cleanup() {
    local exit_code=$?
    echo ""
    echo "🛑 Cleaning up..."

    if [[ "$START_WORKER" == true ]]; then
      pkill -9 -f 'celery -A app.worker' 2>/dev/null
      echo "   Celery worker stopped."
    fi

    if [ -n "$API_PID" ] && kill -0 $API_PID 2>/dev/null; then
        kill $API_PID
        echo "   API process stopped."
    fi

    echo "✅ Cleanup complete."
    exit $exit_code
}

# 捕获信号
trap cleanup EXIT TERM INT

# 启动 Redis（如果需要）
if [[ "$START_REDIS" == true ]]; then
  if pgrep redis-server > /dev/null; then
    echo "✅ Redis is already running."
  else
    redis-server --daemonize yes
    echo "✅ Redis server started."
  fi
else
  echo "⏭️  跳过内置 Redis (使用外部 Redis)"
fi

# 数据库迁移
echo "📦 Upgrading database schema..."
alembic upgrade head

# 启动服务
start_api
start_worker

echo ""
echo "✅ API 服务已启动"
echo "   访问地址: http://0.0.0.0:$PORT"
echo "   健康检查: http://0.0.0.0:$PORT/health"
echo "   API 文档: http://0.0.0.0:$PORT/docs"
echo ""

# 等待 API 进程
wait $API_PID

# 退出
EXIT_STATUS=$?
echo "Gunicorn stopped with exit code $EXIT_STATUS"
exit $EXIT_STATUS

