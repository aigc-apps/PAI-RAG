


## 示例

1. Read命令
```shell
python src/pai_rag/data_ingestion/main.py read --input-path /testdata/testdata/txt --output-path /testdata/output0423/read

```

2. Parse命令
```shell
python src/pai_rag/data_ingestion/main.py parse --input-path /testdata/output0423/read --output-path /testdata/output0423/parse \
    --num-cpus 2 --memory 8
```

3. Split命令
```shell
python src/pai_rag/data_ingestion/main.py split --input-path /testdata/output0423/parse --output-path /testdata/output0423/split \
    --num-cpus 1 --memory 2
```

4. Embed命令
```shell
python src/pai_rag/data_ingestion/main.py embed --input-path /testdata/output0423/split --output-path /testdata/output0423/embed \
    --num-cpus 5 --memory 10 --num-gpus 1 --source huggingface --model bge-m3
```