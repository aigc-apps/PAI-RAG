


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
python src/pai_rag/data_ingestion/main.py split --input-path /testdata/output0423/parse --output-path /testda·ta/output0423/split \
    --num-cpus 1 --memory 2
```

4. Embed命令
```shell
python src/pai_rag/data_ingestion/main.py embed --input-path /testdata/output0423/split --output-path /testdata/output0423/embed \
    --num-cpus 5 --memory 10 --num-gpus 1 --source huggingface --model bge-m3
```


5. Write命令，存入向量数据库
```shell
export PAI_RAG_ENDPOINT=http://127.0.0.1/8680
export PAI_RAG_KEY=12345
python src/pai_rag/data_ingestion/main.py write --input-path /testdata/output0423/embed --num-cpus 10 --memory 20 --rag-endpoint $PAI_RAG_ENDPOINT --rag-key $PAI_RAG_KEY --knowledgebase milvus_test --embed-dims 1024 


```