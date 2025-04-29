


## 示例
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