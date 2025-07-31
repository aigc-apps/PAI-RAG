#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -w workers -p port"
   echo -e "\t-w number of workers, default 1"
   echo -e "\t-p port, default 8680"
   exit 1 # Exit script after printing help
}

while getopts ":w:p:" opt
do
   case "$opt" in
      w ) workers="$OPTARG" ;;
      p ) port="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


# 定义清理函数（在退出或信号捕获时调用）
cleanup() {
    local exit_code=$?
    echo "Cleaning up..."

    pkill -9 -f 'celery -A pairag'
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

echo "Starting celery workers..."

celery -A pairag.mcp.rag.file_worker worker --loglevel=info -c 4 &

echo "Celery is started."


echo "starting web"
cd frontend/
npm install && npm run dev -- --port 3000 &
FRONTEND_PID=$!
echo "frontend is started with pid $FRONTEND_PID."

cd ..

echo "Starting api server..."

workers="${workers:-1}"
port="${port:-8680}"

echo "Starting gunicorn with $workers workers on port $port..."

gunicorn -w $workers -b "0.0.0.0:${port}" -c scripts/gunicorn.conf.py src.pairag.app.app:app --timeout 600
