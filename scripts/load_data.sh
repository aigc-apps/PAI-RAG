#!/bin/bash
set -e


# 获取脚本所在的目录
SCRIPT_DIR=$(dirname "$0")
# 切换到脚本所在目录的上级目录
cd "$SCRIPT_DIR/.."
pwd

python src/pairag/data_ingestion/main.py $*
