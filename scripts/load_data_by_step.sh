#!/bin/bash
set -e


# 修改为你需要的目录
INPUT_PATH="/testdata/testdata/small_txt"
OUTPUT_PATH="/testdata/testdata/output/small0520"


# 获取脚本所在的目录
SCRIPT_DIR=$(dirname "$0")
# 切换到脚本所在目录的上级目录
cd "$SCRIPT_DIR/.."
pwd

echo "reading data source..."
python src/pai_rag/data_ingestion/main.py data-source \
    --enable-delta \
    --input-path $INPUT_PATH \
    --output-path $OUTPUT_PATH

echo "parsing files..."
python src/pai_rag/data_ingestion/main.py parse \
    --input-path $OUTPUT_PATH \
    --output-path $OUTPUT_PATH


echo "splitting into chunks..."
python src/pai_rag/data_ingestion/main.py split \
    --input-path $OUTPUT_PATH \
    --output-path $OUTPUT_PATH \

echo "embedding data..."
python src/pai_rag/data_ingestion/main.py embed \
    --input-path $OUTPUT_PATH \
    --output-path $OUTPUT_PATH \
    --num-gpus 0.5 \
    --num-cpus 6 \
    --memory 16 \
    --concurrency 2


echo "writing to data sink..."
python src/pai_rag/data_ingestion/main.py data-sink \
    --input-path $OUTPUT_PATH \
    --output-path $OUTPUT_PATH \
