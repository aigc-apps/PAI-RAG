# Parameter Configuration Instruction

This guidance primarily walks you through configuring your RAG parameters in **_src/pai_rag/config/settings.toml_**, which is referred to as settings.toml hereafter.

## rag.data_reader

type = [SimpleDirectoryReader, LlamaParseDirectoryReader, DatabaseReader]

SimpleDirectoryReader is the most commonly used data connector. Simply pass in a input directory or a list of files. It will select the best file reader based on the file extensions.

If type = "SimpleDirectoryReader", it is simply as:

    type = "SimpleDirectoryReader"

LlamaParseDirectoryReader is an integration with LlamaParse as the default PDF loader in SimpleDirectoryReader. LlamaParse is an API created by LlamaIndex to efficiently parse and represent files for efficient retrieval and context augmentation using LlamaIndex frameworks.

If type = "LlamaParseDirectoryReader", firstly, you need to login and get an api-key from https://cloud.llamaindex.ai, then you can either fill in "llama_cloud_api_key" in settings.toml as shown below, or set it as an environment variable in the command line via: `export LLAMA_CLOUD_API_KEY="xxx"`

    type = "LlamaParseDirectoryReader"
    llama_cloud_api_key = "your_api_key"

If type = "DatabaseReader", the currently supported database_type is PostgreSQL, you need to specify additional information as shown below:

    type = "DatabaseReader"
    database_type = "PostgreSQL"
    host = "database url"
    port = "datasase port"
    dbname = "target db name"
    user = "username"
    password = "password"

This setting is not available in webui.

## rag.embedding

source = [HuggingFace, OpenAI, DashScope]

Currently, pai_rag supports three embedding sources.

If source = "HuggingFace", you need to further specify model_modelname and embed_batch_size. The default model name and batch size are bge-large-zh-v1.5 and 10, respectively.

    source = "HuggingFace"
    model = "bge-large-zh-v1.5"
    embed_batch_size = 10

Alternatively, if you want to use other huggingface models, please specify parameters as below:

    source = "HuggingFace"
    model = "xxx"
    model_dir = "xxx/xxx"
    embed_batch_size = 20 (for example)

If source = "OpenAI" or "DashScope", you need to provide the corresponding api_key either through setting.toml or via environment variable, also you may specify your preferred batch_size (default=10):

    source = "DashScope" (for example)
    api_key = "xxx"
    embed_batch_size = 10

This setting is also available in webui.

## rag.llm

source = [PaiEas, OpenAI, DashScope]

Currently, pai_rag supports three llm sources.

If source = "PaiEas", you need to specify the following parameters:

    source = "PaiEas"
    endpoint = ""
    token = ""

If source = "OpenAI", gpt-3.5-turbo and gpt-4-turbo are available in webui, also you can specify other versions in setting.toml as below:

    source = "OpenAI"
    model = ""
    temperature = ""

It is noted that **_temperature_** is a parameter ranging from 0 to 1. A high temperature lets the model more creative while a low temperature makes the model more accurate and factual. Its default value in pai_rag is 0.1.

If source = "DashScope", the candidates include qwen-turbo, qwen-max, qwen-plus, qwen-max-1201, qwen-max-longcontext. Parameters can be specified as below:

    source = "DashScope"
    model = ""
    temperature = ""

This setting is also available in webui.

## rag.index

vector_store.type = [FAISS, Hologres, ElasticSearch, AnalyticDB, Milvus]

Currently, pai_rag provides a variety of approaches for creating & storing indices.

If vector_store.type = "FAISS", you can specify a local persist_path as below:

    [rag.index]
    vector_store.type = "FAISS"
    persist_path = "localdata/storage"

If vector_store.type = "Hologres", you need to set up the following information:

    [rag.index]
    persist_path = "localdata/storage"

    [rag.index.vector_store]
    type = "Hologres"
    host = "your hologres url"
    port = 80
    user = "your user name"
    password = "your pwd"
    database = "pairag" (just for example)
    table_name = "pairag"

If vector_store.type = "ElasticSearch", you need to prepare the following information:

    [rag.index]
    persist_path = "localdata/storage"

    [rag.index.vector_store]
    type = "ElasticSearch"
    es_index = "create your index name"
    es_url = "es_host:es_port(9200)"
    es_user = ""
    es_password = ""

If vector_store.type = "AnalyticDB", you need to fill in the following information:

    [rag.index]
    persist_path = "localdata/storage"

    [rag.index.vector_store]
    type = "AnalyticDB"
    ak = ""
    sk = ""
    region_id = ""
    instance_id = ""
    account = ""
    account_password = ""
    namespace = "pairag"
    collection = "pairag_collection"

If vector_store.type = "Milvus", you need to provide the following information:

    [rag.index]
    persist_path = "localdata/storage"

    [rag.index.vector_store]
    type = "Milvus"
    host = ""
    port = ""
    user = ""
    password = ""
    database = "pairag"
    collection = "pairag_collection"

This setting is also available in webui.

## rag.node_parser

type = [Token, Sentence, SentenceWindow, Semantic]

Currently, pai_rag supports four node_parsers.

If type = "Token", it attempts to split to a consistent chunk size according to raw token counts. You can refer to the following configuration:

    type = "Token"
    chunk_size = 1024
    chunk_overlab = 128

If type = "Sentence", it attempts to split text while respecting the boundaries of sentences. You can refer to the following setting:

    type = "Sentence"
    chunk_size = 500
    chunk_overlap = 10

If type = "SentenceWindow", it splits all documents into individual sentences. The resulting nodes also contain the surrounding "window" of sentences around each node in the metadata. This is most useful for generating embeddings that have a very specific scope. Parameters are to be set as follows:

    type = "SentenceWindow"
    chunk_size = 500
    chunk_overlap = 10
    paragraph_separator = "\n\n\n"
    window_size = 3

If type = "Semantic", it adaptively picks the breakpoint in-between sentences using embedding similarity instead of chunking text with a fixed chunk size. This ensures that a "chunk" contains sentences that are semantically related to each other. Parameters are to be set as below:

    type = "Semantic"
    breakpoint_percentile_threshold = 95
    buffer_size = 1

Among the different parameters, chunk_size and chunk_overlap can also be adjusted in webui.

## rag.retriever

retrieval_mode = [hybrid, embedding, keyword]

"keyword" denotes using keyword only for sparse retrieval; "embedding" represents using embedded vector for dense retrieval; "hybrid" means using both.

**If "ElasticSearch" is selected in rag.index**, you will be using the built-in retriever in ElasticSearch and the configuration is as follows:

    retrieval_mode = ""  # one of keyword, embedding and hybrid
    similarity_top_k = 3

If "ElasticSearch" is not selected previously and retrieval_mode = "keyword" or "embedding", the setting is as below:

    retrieval_mode = ""  # one of keyword and embedding
    similarity_top_k = 3

If "ElasticSearch" is not selected previously and retrieval_mode = "hybrid", the setting can be referred to as following:

    retrieval_mode = "hybrid"
    similarity_top_k = 3
    BM25_weight = 0.5
    vector_weight = 0.5
    fusion_mode = "reciprocal_rerank"   # [simple, reciprocal_rerank, dist_based_score, relative_score]
    query_rewrite_n = 1         # set to 1 to disable query generation

Herein, `similarity_top_k` is used for all retrievals, each retrieval outputs the same number of top_k nodes and the final fusion retrieval further selects top_k nodes among the previous selected nodes.
Each retrieval can be weighted depending on specific usage scenarios.
`fusion_mode` has four candidates with different reranking algorithm, e.g. "simple" means simple re-ordering of results based on original scores. "reciprocal_rerank" is the default setting.
`query_rewrite_n` denotes the number of queries to generate, e.g. to generate 2 more queries, this parameter can be set to 3.

The retrieval mode as well as the similarity top k can also be set in webui.

## rag.postprocessor

rerank_model = [no-reranker, bge-reranker-base, bge-reranker-large, llm-reranker]

Currently, pai_rag supports three rerank models, among which llm-reranker uses the llm itself.

If you do not need a reranker, simply set as follows:

    rerank_model = "no_reranker"

If a candidate model is selected, please also specify the top number of nodes to be returned:

    rerank_model = ""  # [bge-reranker-base, bge-reranker-large, llm-reranker]
    top_n = 2

This setting is also available in webui.

## rag.synthesizer

type = [Refine, Compact, TreeSummarize, SimpleSummarize]

Currently, pai_rag supports four synthesizers.

`Refine`: create and refine an answer by sequentially going through each retrieved text chunk. This makes a separate LLM call per Node/retrieved chunk.
`Compat`: similar to refine but compact (concatenate) the chunks beforehand, resulting in less LLM calls.
`TreeSummarize`: Query the LLM as many times as needed so that all concatenated chunks have been queried, resulting in as many answers that are themselves recursively used as chunks in a tree_summarize LLM call and so on, until there's only one chunk left, and thus only one final answer.
`SimpleSummarize`: Truncates all text chunks to fit into a single LLM prompt. Good for quick summarization purposes, but may lose detail due to truncation.

## rag.query_engine

type = "RetrieverQueryEngine"

Query engine is a generic interface that allows you to ask question over your data. It takes in a natural language query, and returns a rich response.

## rag.chat_store

type = [Local, Aliyun-Redis]

Store you chat history locally or persist it in Aliyun-Redis.

If type = "Local", persist_path can be set as below as default, or you can specify a preferred destination.

    type = "Local"
    persist_path = "localdata/storage"

If type = "Aliyun-Redis", you need to get access to the your remote redis by fulfilling host and password as shown below

    type = "Aliyun-Redis"
    host = "Aliyun-Redis host"
    password = "Aliyun-Redis user:pwd"

This setting is not available in webui.

## rag.agent

type = [react]

A ReACT agent is a technique that combines the reasoning of LLMs with actionable steps to create a more sophisticated system.

Currently, pai_rag supports the react type with googlesearch and calculator tools with add, subtract, multiply and divide functions. This setting is not available in webui.

## rag.tool

type = [calculator, googlewebsearch]

If type = "googlewebsearch", you need to specify parameters as follows:

    type = "googlewebsearch"
    google_search_api = ""
    google_search_engine = ""

If type = "calculator", no more parameters need to be specified:

    type = "calculator"

You can define more customized tools in `src/pai_rag/modules/tool/`.
