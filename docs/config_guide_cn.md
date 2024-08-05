# 参数配置说明

这篇说明主要指导如何在 src/pai_rag/config/settings.toml 中配置您的 RAG 参数，以上路径在下文中简称 settings.toml

## rag.data_reader

type = [SimpleDirectoryReader, LlamaParseDirectoryReader, DatabaseReader]

SimpleDirectoryReader是最常用的数据连接器。只需传入一个输入目录或文件列表。它会根据文件扩展名选择最合适的文件阅读器。

如果 type = "SimpleDirectoryReader", 无需额外参数:

    type = "SimpleDirectoryReader"

LlamaParseDirectoryReader是SimpleDirectoryReader中集成LlamaParse的PDF加载器。LlamaParse是由LlamaIndex创建的API，能够高效解析和表示文件，以利用LlamaIndex框架进行高效的检索和上下文增强。

如果 type = "LlamaParseDirectoryReader", 首先，您需要登录并从 https://cloud.llamaindex.ai 获取一个 API 密钥。然后，您可以按照下面所示，在 settings.toml 文件中填写 "llama_cloud_api_key"，或者通过命令行将其设置为环境变量：`export LLAMA_CLOUD_API_KEY="xxx"`

    type = "LlamaParseDirectoryReader"
    llama_cloud_api_key = "your_api_key"

如果 type = "DatabaseReader", 目前支持的数据库类型是PostgreSQL，需要按照下面所示补充额外信息：

    type = "DatabaseReader"
    database_type = "PostgreSQL"
    host = "database url"
    port = "datasase port"
    dbname = "target db name"
    user = "username"
    password = "password"

该设置在网页（webui）中不可用。

## rag.embedding

source = [HuggingFace, OpenAI, DashScope]

目前, pai_rag 支持以上三种 embedding 源.

如果 source = "HuggingFace", 您需要进一步指定 model_name 和 embed_batch_size。默认的模型名称和批处理大小分别为 bge-small-zh-v1.5 和 10。

    source = "HuggingFace"
    model_name = "bge-small-zh-v1.5"
    embed_batch_size = 10

或者, 如果你想使用其它 huggingface 模型, 请指定如下参数：

    source = "HuggingFace"
    model_name = "xxx"
    model_dir = "xxx/xxx"
    embed_batch_size = 20 (for example)

如果 source = "OpenAI" or "DashScope", 您需要通过设置 settings.toml 或环境变量提供相应的 api_key，且需要指定批处理大小：
source = "DashScope" (for example)
api_key = "xxx"
embed_batch_size = 10

该设置在网页中不可用。

## rag.llm

source = [PaiEas, OpenAI, DashScope]

目前, pai_rag 支持以上三类大语言模型.

如果 source = "PaiEas", 需要指定参数如下:

    source = "PaiEas"
    endpoint = ""
    token = ""

如果 source = "OpenAI", 网页中可选 gpt-3.5-turbo 和 gpt-4-turbo，您也可以在 setting.toml 中指定其它版本：

    source = "OpenAI"
    model = ""
    temperature = ""

**_temperature_** 是一个范围从 0 到 1 的参数。设置较高值可以让模型更具创造性，而较低值可以使模型更准确和事实导向。在 pai_rag 中，其默认值为 0.1。

如果 source = "DashScope", 可选模型包括 qwen-turbo, qwen-max, qwen-plus, qwen-max-1201, qwen-max-longcontext。参数设置如下：

    source = "DashScope"
    model = ""
    temperature = ""

该设置在网页中不可用。

## rag.index

vector_store.type = [FAISS, Hologres, ElasticSearch, AnalyticDB, Milvus]

目前, pai_rag 支持多种方式创建和存储索引。

如果 vector_store.type = "FAISS", 直接在[rag.index]中指定一个持久化路径：

    [rag.index]
    vector_store.type = "FAISS"
    persist_path = "localdata/storage"

如果 vector_store.type = "Hologres", 配置如下：

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

如果 vector_store.type = "ElasticSearch", 配置如下：

    [rag.index]
    persist_path = "localdata/storage"

    [rag.index.vector_store]
    type = "ElasticSearch"
    es_index = "create your index name"
    es_url = "es_host:es_port(9200)"
    es_user = ""
    es_password = ""

如果 vector_store.type = "AnalyticDB", 需要补充如下信息：

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

如果 vector_store.type = "Milvus", 需要提供如下信息：

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

该设置也可在网页中配置。

## rag.node_parser

type = [Token, Sentence, SentenceWindow, Semantic]

目前, pai_rag 支持四种 node_parser.

如果 type = "Token", 会尝试根据原始标记（token）数量将其分割为一致的块大小，配置如下：

    type = "Token"
    chunk_size = 1024
    chunk_overlab = 128

如果 type = "Sentence", 会尝试在维持句子边界的基础上分割文本，配置如下：

    type = "Sentence"
    chunk_size = 500
    chunk_overlap = 10

如果 type = "SentenceWindow", 将所有文档分割成单独的句子。生成的节点还包含每个节点周围句子的“窗口”信息作为元数据。参数设置如下：

    type = "SentenceWindow"
    chunk_size = 500
    chunk_overlap = 10
    paragraph_separator = "\n\n\n"
    window_size = 3

如果 type = "Semantic", 通过嵌入相似性自适应地选择句子之间的断点，而不是使用固定的块大小来分割文本。这确保了一个“块”包含语义相关的句子。参数设置如下：

    type = "Semantic"
    breakpoint_percentile_threshold = 95
    buffer_size = 1

以上各种参数中，chunk_size 和 chunk_overlap 可以在网页中设置。

## rag.retriever

retrieval_mode = [hybrid, embedding, keyword]

"keyword" 表示基于关键词的稀疏检索; "embedding" 表示基于相似度的向量检索; "hybrid" 表示混合检索，包括关键词和向量。

**如果在rag.index中配置了 "ElasticSearch"**, 将会使用到es内置的检索器，此时依然支持三种检索模式，需要明确返回相似度排名靠前的节点数量：

    retrieval_mode = ""  # one of keyword, embedding and hybrid
    similarity_top_k = 3

如果在rag.index中未配置 "ElasticSearch" 且 retrieval_mode = "keyword" 或 "embedding", 配置如下：

    retrieval_mode = ""  # one of keyword and embedding
    similarity_top_k = 3

如果在rag.index中未配置 "ElasticSearch" 且 retrieval_mode = "hybrid", 配置如下：

    retrieval_mode = "hybrid"
    similarity_top_k = 3
    BM25_weight = 0.5
    vector_weight = 0.5
    fusion_mode = "reciprocal_rerank"   # [simple, reciprocal_rerank, dist_based_score, relative_score]
    query_rewrite_n = 1         # set to 1 to disable query generation

这里, `similarity_top_k` 用于所有检索器，每个检索器返回相同数量的前 k 个节点，最终的融合检索器在所有召回节点中再筛选前 k 个节点。 每个检索器支持分配权重。`fusion_mode` 包含四种重排放法，例如，“simple”表示根据原始节点分数简单重新排序，其它三种可根据名称简单理解，其中，“reciprocal_rerank”是默认设置。`query_rewrite_n` 表示要重写的查询数量。例如，要生成2个查询，则可将此参数设为3。

retrieval_mode 和 similarity_top_k 可在网页中设置。

## rag.postprocessor

rerank_model = [no-reranker, bge-reranker-base, bge-reranker-large, llm-reranker]

目前, pai_rag 支持三种重排模型, 其中 llm-reranker 使用 llm 自身能力进行重排，其它两种为专门训练的小模型。

如果无需使用reranker，配置如下:

    rerank_model = "no_reranker"

如果选择其中一个模型, 需要指定重排后返回的节点数量:

    rerank_model = ""  # [bge-reranker-base, bge-reranker-large, llm-reranker]
    top_n = 2

以上参数支持网页中配置。

## rag.synthesizer

type = [Refine, Compact, TreeSummarize, SimpleSummarize]

目前, pai_rag 支持四种合成器.

`Refine`: 通过逐个处理每个检索到的文本块来创建和完善答案。这会为每个节点/检索到的块单独调用llm。
`Compat`: 类似于 Refine，但是会预先将文本块连接在一起，从而减少llm的调用次数。
`TreeSummarize`: 根据需要多次查询llm，使所有连接在一起的块都被查询，多个答案本身会递归地作为树状总结调用的块，依此类推，直到只剩下一个块，从而产出一个最终答案。
`SimpleSummarize`: 将所有文本块截断以适应单个llm提示。适用于快速总结目的，但由于截断可能会丢失细节。

## rag.query_engine

type = "RetrieverQueryEngine"

查询引擎（query engine）是一个通用接口，接收自然语言查询，并返回丰富的响应。

## rag.llm_chat_engine

type = "SimpleChatEngine"

基于 query engine 之上的一个高级接口，用于与数据进行对话（而不是单一的问答），可类比为状态化的查询引擎。

## rag.chat_engine

type = [CondenseQuestionChatEngine]

Condense question 是建立在查询引擎query engine之上的简易聊天模式。每次聊天交互中：首先从对话上下文和最后一条消息生成一个独立的问题，然后用这个简化的问题查询查询引擎以获取回复。

## rag.chat_store

type = [Local, Aliyun-Redis]

本地或redis存储聊天记录。

如果 type = "Local", 默认的持久化路径设置如下, 或者自定义路径。

    type = "Local"
    persist_path = "localdata/storage"

如果 type = "Aliyun-Redis", 你需要填写如下的主机和密码访问你的远程 Redis

    type = "Aliyun-Redis"
    host = "Aliyun-Redis host"
    password = "Aliyun-Redis user:pwd"

该设置在网页中不可用。

## rag.evaluation

目前, pai_rag 支持对检索效果和回复效果的评估。

检索器的评估指标包括 "mrr"（平均倒数排名） 和 "hit_rate"（命中率）, 可以指定两者或者任意一个：

    retrieval = ["mrr", "hit_rate"]

对最终回复的评估指标包括 "Faithfulness"（事实性）, "Answer Relevancy"（回答相关性）, "Correctness"（正确性）以及 "Semantic Similarity"（语义相似性）, 可以选择全部或者任意若干：

    response = ["Faithfulness", "Answer Relevancy", "Correctness", "Semantic Similarity"]

说明：这些评估需要基于特定的数据集格式，可以通过`src/pai_rag/evaluations/batch_evaluator`中的功能生成, 也可以提供正确格式的数据集路径：

     qa_dataset_path = ""

该设置在网页中不可用。

## rag.agent

type = [react]

通过交替进行reasoning和acting的方式帮助智能体更好地应对和解决复杂问题。相较于一般的reasoning方法，能够根据当前的场景进行分析，做出更具有针对性的行动，并且提供了更好的可解释性和更高的容错度。该设置在网页中不可用。

## rag.tool

type = [calculator, googlewebsearch]

配合agent使用，目前内置了 googlesearch 和 calculator 工具。

如果 type = "googlewebsearch", 配置如下:

    type = "googlewebsearch"
    google_search_api = ""
    google_search_engine = ""

如果 type = "calculator", 无需其它配置:

    type = "calculator"

可在`src/pai_rag/modules/tool/`中自定义工具。
