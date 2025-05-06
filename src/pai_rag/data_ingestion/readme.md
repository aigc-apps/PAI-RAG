## 分步骤执行示例

设置output_path，每次执行选择一个新路径，做好数据隔离

```shell
export OUTPUT_PATH=/testdata/output0428
```

1. Read命令

```shell
python src/pai_rag/data_ingestion/main.py read --input-path /testdata/testdata/txt --output-path $OUTPUT_PATH/read --enable-delta

```

2. Parse命令

```shell
python src/pai_rag/data_ingestion/main.py parse --input-path $OUTPUT_PATH/read --output-path $OUTPUT_PATH/parse \
    --num-cpus 2 --memory 8
```

3. Split命令

```shell
python src/pai_rag/data_ingestion/main.py split --input-path $OUTPUT_PATH/parse --output-path $OUTPUT_PATH/split \
    --num-cpus 1 --memory 2
```

4. Embed命令

```shell
python src/pai_rag/data_ingestion/main.py embed --input-path $OUTPUT_PATH/split --output-path $OUTPUT_PATH/embed \
    --num-cpus 5 --memory 10 --num-gpus 1 --source huggingface --model bge-m3
```

5. Write命令，存入向量数据库

```shell
python src/pai_rag/data_ingestion/main.py write --input-path $OUTPUT_PATH/embed --num-cpus 10 --memory 20
```

## E2E执行示例

设置output_path，每次执行选择一个新路径，做好数据隔离

```shell
export OUTPUT_PATH=/testdata/output0505
export PAI_RAG_ENDPOINT=
export PAI_RAG_KEY=

python src/pai_rag/data_ingestion/main.py e2e --input-path xx --output-path $OUTPUT_PATH
```

### Appendix

1. 安装Langstudio依赖

```sh
poetry add llama-index-vector-stores-hologres
poetry add llama-index-vector-stores-dashvector
poetry add promptflow-tracing https://pai-sdk.oss-cn-shanghai.aliyuncs.com/promptflow-tracing/dist/promptflow_tracing-1.17.0%2Blangstudio2502.378942e-py3-none-any.whl
poetry add promptflow-core https://pai-sdk.oss-cn-shanghai.aliyuncs.com/promptflow-core/dist/promptflow_core-1.17.0%2Blangstudio2502.378942e-py3-none-any.whl
poetry add promptflow-devkit https://pai-sdk.oss-cn-shanghai.aliyuncs.com/promptflow-devkit/dist/promptflow_devkit-1.17.0%2Blangstudio2502.378942e-py3-none-any.whl
poetry add promptflow https://pai-sdk.oss-cn-shanghai.aliyuncs.com/promptflow/dist/promptflow-1.17.0%2Blangstudio2502.378942e-py3-none-any.whl
poetry add https://pai-sdk.oss-cn-shanghai.aliyuncs.com/langstudio/dist/langstudio-0.2.0.dev8-py3-none-any.whl
```
