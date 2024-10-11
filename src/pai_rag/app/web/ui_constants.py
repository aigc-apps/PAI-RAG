from pai_rag.integrations.synthesizer.pai_synthesizer import (
    DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
)
from pai_rag.utils.prompt_template import DEFAULT_TEXT_QA_PROMPT_TMPL

DEFAULT_TEXT_QA_PROMPT_TMPL = DEFAULT_TEXT_QA_PROMPT_TMPL
DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL = DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL

DA_GENERAL_PROMPTS = "给定一个输入问题，创建一个语法正确的{dialect}查询语句来执行，不要从特定的表中查询所有列，只根据问题查询几个相关的列。请注意只使用你在schema descriptions 中看到的列名。\n=====\n 小心不要查询不存在的列。请注意哪个列位于哪个表中。必要时，请使用表名限定列名。\n=====\n 你必须使用以下格式，每项占一行：\n\n Question: Question here\n SQLQuery: SQL Query to run \n\n Only use tables listed below.\n {schema}\n\n Question: {query_str} \n SQLQuery: "
DA_SQL_PROMPTS = "给定一个输入问题，其中包含了需要执行的SQL语句，请提取问题中的SQL语句，并使用{schema}进行校验优化，生成符合相应语法{dialect}和schema的SQL语句。\n=====\n 你必须使用以下格式，每项占一行：\n\n Question: Question here\n SQLQuery: SQL Query to run \n\n Only use tables listed below.\n {schema}\n\n Question: {query_str} \n SQLQuery: "


# WELCOME_MESSAGE = """
#             # \N{fire} Chatbot with RAG on PAI !
#             ### \N{rocket} Build your own personalized knowledge base question-answering chatbot.

#             #### \N{fire} Platform: [PAI](https://help.aliyun.com/zh/pai)  /  [PAI-EAS](https://www.aliyun.com/product/bigdata/learn/eas)  / [PAI-DSW](https://pai.console.aliyun.com/notebook) &emsp;  \N{rocket} Supported VectorStores:  [Milvus](https://www.aliyun.com/product/bigdata/emapreduce/milvus) / [Hologres](https://www.aliyun.com/product/bigdata/hologram)  /  [ElasticSearch](https://www.aliyun.com/product/bigdata/elasticsearch)  /  [AnalyticDB](https://www.aliyun.com/product/apsaradb/gpdb)  /  [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss) / [OpenSearch](https://help.aliyun.com/zh/open-search/vector-search-edition/product-overview/)

#             #### \N{fire} <a href='/docs'>API Docs</a> &emsp; \N{rocket} \N{fire}  欢迎加入【PAI】RAG答疑群 27370042974
#             """

WELCOME_MESSAGE = """
            # \N{fire} PAI-RAG Dashboard

            #### \N{rocket} Join the DingTalk Q&A Group: 27370042974
            """

DEFAULT_CSS_STYPE = """
        h1, h3, h4 {
            text-align: center;
            display:block;
        }
        """

DEFAULT_EMBED_SIZE = 1024

DEFAULT_HF_EMBED_MODEL = "bge-large-zh-v1.5"


EMBEDDING_MODEL_DEPRECATED = [
    "bge-small-zh-v1.5",
    "SGPT-125M-weightedmean-nli-bitfit",
    "text2vec-large-chinese",
    "paraphrase-multilingual-MiniLM-L12-v2",
]

EMBEDDING_MODEL_LIST = [
    "bge-large-zh-v1.5",
    "Chuxin-Embedding",
    "bge-large-en-v1.5",
    "gte-large-en-v1.5",
    "bge-m3",
    "multilingual-e5-large-instruct",
]

EMBEDDING_DIM_DICT = {
    "bge-large-zh-v1.5": 1024,
    "Chuxin-Embedding": 1024,
    "bge-large-en-v1.5": 1024,
    "gte-large-en-v1.5": 1024,
    "bge-m3": 1024,
    "multilingual-e5-large-instruct": 1024,
    "bge-small-zh-v1.5": 512,
    "SGPT-125M-weightedmean-nli-bitfit": 768,
    "text2vec-large-chinese": 1024,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}

EMBEDDING_TYPE_DICT = {
    "bge-large-zh-v1.5": "Chinese",
    "Chuxin-Embedding": "Chinese",
    "bge-large-en-v1.5": "English",
    "gte-large-en-v1.5": "English",
    "bge-m3": "Multilingual",
    "multilingual-e5-large-instruct": "Multilingual",
    "bge-small-zh-v1.5": "Chinese",
    "SGPT-125M-weightedmean-nli-bitfit": "Multilingual",
    "text2vec-large-chinese": "Chinese",
    "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual",
}

EMBEDDING_MODEL_LINK_DICT = {
    "bge-large-zh-v1.5": "https://huggingface.co/BAAI/bge-large-zh-v1.5",
    "Chuxin-Embedding": "https://huggingface.co/chuxin-llm/Chuxin-Embedding",
    "bge-large-en-v1.5": "https://huggingface.co/BAAI/bge-large-en-v1.5",
    "gte-large-en-v1.5": "https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5",
    "bge-m3": "https://huggingface.co/BAAI/bge-m3",
    "multilingual-e5-large-instruct": "https://huggingface.co/intfloat/multilingual-e5-large-instruct",
    "bge-small-zh-v1.5": "https://huggingface.co/BAAI/bge-small-zh-v1.5",
    "SGPT-125M-weightedmean-nli-bitfit": "https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    "text2vec-large-chinese": "https://huggingface.co/GanymedeNil/text2vec-large-chinese",
    "paraphrase-multilingual-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

EMBEDDING_API_KEY_DICT = {"HuggingFace": False, "OpenAI": True, "DashScope": True}

LLM_MODEL_KEY_DICT = {
    "DashScope": [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen-max-1201",
        "qwen-max-longcontext",
    ],
    "OpenAI": [
        "gpt-3.5-turbo",
        "gpt-4-turbo",
    ],
}

MLLM_MODEL_KEY_DICT = {
    "DashScope": [
        "qwen-vl-max",
        "qwen-vl-turbo",
    ]
}

EMPTY_KNOWLEDGEBASE_MESSAGE = "We couldn't find any documents related to your question: {query_str}. \n\n You may try lowering the similarity_threshold or uploading relevant knowledge files."
