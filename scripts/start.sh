#!/bin/bash

# ===================================================================
# 启动脚本：支持可配置参数
# 支持参数：
#   --frontend-port       前端端口 (默认: 3000)
#   --backend-port        后端端口 (默认: 8000)
#   --api-instances       API 实例数量 (默认: 1)
#   --worker-instances    Worker 实例数量 (默认: 1)
#   --help                显示帮助
# ===================================================================

# 默认值
FRONTEND_PORT=${FRONTEND_PORT:-8680}
BACKEND_PORT=${BACKEND_PORT:-8688}
API_INSTANCE_COUNT=${API_INSTANCE_COUNT:-1}
WORKER_INSTANCE_COUNT=${WORKER_INSTANCE_COUNT:-2}


# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --frontend-port)
      FRONTEND_PORT="$2"
      if ! [[ "$FRONTEND_PORT" =~ ^[0-9]+$ ]] || [ "$FRONTEND_PORT" -lt 1 ] || [ "$FRONTEND_PORT" -gt 65535 ]; then
        echo "错误: --frontend-port 必须是 1-65535 之间的有效端口号"
        exit 1
      fi
      shift 2
      ;;
    --backend-port)
      BACKEND_PORT="$2"
      if ! [[ "$BACKEND_PORT" =~ ^[0-9]+$ ]] || [ "$BACKEND_PORT" -lt 1 ] || [ "$BACKEND_PORT" -gt 65535 ]; then
        echo "错误: --backend-port 必须是 1-65535 之间的有效端口号"
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
    --worker-instances)
      WORKER_INSTANCE_COUNT="$2"
      if ! [[ "$WORKER_INSTANCE_COUNT" =~ ^[0-9]+$ ]] || [ "$WORKER_INSTANCE_COUNT" -lt 0 ]; then
        echo "错误: --worker-instances 必须是非负整数"
        exit 1
      fi
      shift 2
      ;;
    --help|-h)
      echo "用法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --frontend-port PORT       前端服务端口 (默认: 3000)"
      echo "  --backend-port PORT        后端服务端口 (默认: 8000)"
      echo "  --api-instances N          启动 N 个 API 实例 (默认: 1)"
      echo "  --worker-instances N       启动 N 个 Worker 实例 (默认: 1)"
      echo "  --help                     显示此帮助信息"
      echo ""
      echo "示例:"
      echo "  $0 --frontend-port 3001 --backend-port 8080 --api-instances 2 --worker-instances 3"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      echo "使用 --help 查看帮助"
      exit 1
      ;;
  esac
done

NEXT_PUBLIC_API_BASE="http://localhost:${BACKEND_PORT}"

# ===================================================================
# 开始启动服务
# ===================================================================

echo "🚀 启动服务配置："
echo "   前端端口: $FRONTEND_PORT"
echo "   后端端口: $BACKEND_PORT"
echo "   API 实例数: $API_INSTANCE_COUNT"
echo "   Worker 实例数: $WORKER_INSTANCE_COUNT"
echo "   NEXT_PUBLIC_API_BASE: $NEXT_PUBLIC_API_BASE"
echo
echo "🚀 启动服务中..."


# 启动前端（假设使用 Vite/React）
start_frontend() {
  echo "👉 启动前端服务 on port $FRONTEND_PORT"
  cd frontend || { echo "错误: 找不到 frontend 目录"; exit 1; }
  npm install || { echo "错误: npm 安装失败"; exit 1; }
  NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE npm run dev -- --port $FRONTEND_PORT &
  FRONTEND_PID=$!
  cd ..
}


# 启动 API 实例（假设使用 FastAPI/Flask）
start_api() {
  echo "👉 启动 API 实例 on port $BACKEND_PORT"
  # 示例：uvicorn app:app --port $port
  gunicorn -w $API_INSTANCE_COUNT -b "0.0.0.0:${BACKEND_PORT}" -c scripts/gunicorn.conf.py app.app:app --timeout 600
  API_PIDS[$instance]=$!
}

# 启动 Worker 实例
start_worker() {
  echo "👉 启动 Celery Worker"
  # 示例：python worker.py
  celery -A app.worker worker --loglevel=info -c 4 &
  echo "Celery已启动."
}

# 定义清理函数（在退出或信号捕获时调用）
cleanup() {
    local exit_code=$?
    echo "Cleaning up..."

    pkill -9 -f 'celery -A app.worker'
    echo "celery job stopped."

    echo "Script exited with code $exit_code."
    exit "$exit_code"

   echo "Cleaning up frontend process..."
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo "frontend process stopped"
    else
        echo "frontend process already stopped"
    fi

}

# 捕获信号（SIGTERM, SIGINT, EXIT）
trap cleanup EXIT TERM INT

# 检查Redis服务是否已经在运行
if pgrep redis-server > /dev/null
then
   echo "Redis is already running."
else
   redis-server &
   echo "Starting redis server."
fi

start_frontend

start_worker

start_api
