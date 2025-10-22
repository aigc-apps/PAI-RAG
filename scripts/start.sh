#!/bin/bash

# ===================================================================
# 启动脚本：支持可配置参数
# 支持参数：
#   --port                应用入口 (默认: 8680)
#   --frontend-port       前端端口 (默认: 8681)
#   --backend-port        后端端口 (默认: 8682)
#   --api-instances       API 实例数量 (默认: 1)
#   --worker-instances    Worker 实例数量 (默认: 1)
#   --dev                 Develop模式（不开启nginx转发）
#   --help                显示帮助
# ===================================================================

# 默认值
PORT=${PORT:-8680}
FRONTEND_PORT=${FRONTEND_PORT:-8681}
BACKEND_PORT=${BACKEND_PORT:-8682}
API_INSTANCE_COUNT=${API_INSTANCE_COUNT:-1}
WORKER_INSTANCE_COUNT=${WORKER_INSTANCE_COUNT:-2}
DEV_MODE=${DEV_MODE:-false}


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
    --port)
      PORT="$2"
      if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
        echo "错误: --port 必须是 1-65535 之间的有效端口号"
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
    --worker-instances|-w)
      WORKER_INSTANCE_COUNT="$2"
      if ! [[ "$WORKER_INSTANCE_COUNT" =~ ^[0-9]+$ ]] || [ "$WORKER_INSTANCE_COUNT" -lt 0 ]; then
        echo "错误: --worker-instances 必须是非负整数"
        exit 1
      fi
      shift 2
      ;;
    --dev)
      DEV_MODE=true
      shift 1
      ;;
    --help|-h)
      echo "用法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --port PORT                应用端口    (默认: 8680) dev模式为前端端口，PRODUCTIoN模式为NGINX端口"
      echo "  --frontend-port PORT       前端服务端口 (默认: 8681), 仅PRODUCTION模式生效"
      echo "  --backend-port PORT        后端服务端口 (默认: 8682)"
      echo "  --api-instances N          启动 N 个 API 实例 (默认: 1)"
      echo "  --worker-instances N       启动 N 个 Worker 实例 (默认: 1)"
      echo "  --dev                      使用dev模式启动web, 将不会配置nginx"
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

# ===================================================================
# 开始启动服务
# ===================================================================

echo "🚀 启动服务配置："
echo "   应用端口: $PORT 前端端口: $FRONTEND_PORT"
echo "   后端端口: $BACKEND_PORT"
echo "   API 实例数: $API_INSTANCE_COUNT"
echo "   Worker 实例数: $WORKER_INSTANCE_COUNT"
echo "   开发模式: $DEV_MODE"
echo
echo "🚀 启动服务中..."


setup_nginx() {
  echo "🚀 配置 Nginx..."
  # 模板路径
  TEMPLATE="./scripts/nginx.template.conf"
  CONFIG="/etc/nginx/conf.d/pairag.conf"

  # 替换变量并生成实际配置
  echo "Generating Nginx config with:"

  export FRONTEND_PORT=$FRONTEND_PORT
  export BACKEND_PORT=$BACKEND_PORT
  export PORT=$PORT

  envsubst < "$TEMPLATE" | sed -e 's/§/$/g' > "$CONFIG"

  # 测试配置
  nginx -t
  if [ $? -ne 0 ]; then
    echo "❌ Nginx config test failed"
    exit 1
  fi

  # 重新加载 Nginx
  service nginx start
  service nginx reload
  echo "✅ Nginx reloaded with new ports"
}


# 启动前端（假设使用 Vite/React）
start_frontend() {
  cd frontend || { echo "错误: 找不到 frontend 目录"; exit 1; }

  if [[ "$DEV_MODE" == false ]]; then
    echo "👉 启动前端服务 on port $FRONTEND_PORT"
    export FRONTEND_PORT=$FRONTEND_PORT
    NEXT_PUBLIC_DEVELOP_MODE=false NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:$BACKEND_PORT npm run start  -- --port $FRONTEND_PORT &
  else
    npm install || { echo "错误: npm 安装失败"; exit 1; }
    echo "👉 开发模式启动前端服务 on port $PORT"
    NEXT_PUBLIC_DEVELOP_MODE=true NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:$BACKEND_PORT npm run dev -- --port $PORT &
  fi
  FRONTEND_PID=$!
  echo "👉 启动前端服务 with pid $FRONTEND_PID."
  cd ..
}


# 启动 API 实例（假设使用 FastAPI/Flask）
start_api() {
  echo "👉 启动 API 实例 on port $BACKEND_PORT"
  # 示例：uvicorn app:app --port $port
  export BACKEND_PORT=$BACKEND_PORT
  gunicorn -w $API_INSTANCE_COUNT -b "0.0.0.0:${BACKEND_PORT}" -c scripts/gunicorn.conf.py app.main:app --timeout 600 &
  API_PID=$!
}

# 启动 Worker 实例
start_worker() {
  echo "👉 启动 Celery Worker"
  # 示例：python worker.py
  celery -A app.worker worker --loglevel=info -c $WORKER_INSTANCE_COUNT &
  echo "Celery已启动."
}

# 定义清理函数（在退出或信号捕获时调用）
cleanup() {
    local exit_code=$?
    echo "Cleaning up..."

    pkill -9 -f 'celery -A app.worker'
    echo "celery job stopped."

   echo "Cleaning up frontend process..."
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo "frontend process stopped"
    else
        echo "frontend process already stopped"
    fi

    echo "Cleaning up API process..."
    kill $API_PID
    echo "API process killed"
}

# 捕获信号（SIGTERM, SIGINT, EXIT）
trap cleanup EXIT TERM INT

if  [[ "$DEV_MODE" == false ]]; then
  setup_nginx
fi

# 检查Redis服务是否已经在运行
if pgrep redis-server > /dev/null
then
   echo "Redis is already running."
else
   redis-server &
   echo "Starting redis server."
fi

echo "upgrading db schema.."
alembic upgrade head

start_api

start_frontend

start_worker

wait $API_PID

# Exit with the same code as Gunicorn
EXIT_STATUS=$?
echo "Gunicorn stopped with exit code $EXIT_STATUS"
exit $EXIT_STATUS
