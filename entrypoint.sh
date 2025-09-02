#!/bin/sh
echo "Entrypoint: $@"

# 用 exec 接管进程，运行传入的命令，用于容器内安全退出
exec "$@"

