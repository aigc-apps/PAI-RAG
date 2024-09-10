SIMPLE_PROMPTS = "参考内容如下：\n{context_str}\n作为个人知识答疑助手，请根据上述参考内容回答下面问题，答案中不允许包含编造内容。\n用户问题:\n{query_str}"
GENERAL_PROMPTS = '基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context_str}\n=====\n用户问题:\n{query_str}'
EXTRACT_URL_PROMPTS = "你是一位智能小助手，请根据下面我所提供的相关知识，对我提出的问题进行回答。回答的内容必须包括其定义、特征、应用领域以及相关网页链接等等内容，同时务必满足下方所提的要求！\n=====\n 知识库相关知识如下:\n{context_str}\n=====\n 请根据上方所提供的知识库内容与要求，回答以下问题:\n {query_str}"
ACCURATE_CONTENT_PROMPTS = "你是一位知识小助手，请根据下面我提供的知识库中相关知识，对我提出的若干问题进行回答，同时回答的内容需满足我所提的要求! \n=====\n 知识库相关知识如下:\n{context_str}\n=====\n 请根据上方所提供的知识库内容与要求，回答以下问题:\n {query_str}"

PROMPT_MAP = {
    SIMPLE_PROMPTS: "Simple",
    GENERAL_PROMPTS: "General",
    EXTRACT_URL_PROMPTS: "Extract URL",
    ACCURATE_CONTENT_PROMPTS: "Accurate Content",
}

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

DEFAULT_EMBED_SIZE = 1536

EMBEDDING_DIM_DICT = {
    "bge-small-zh-v1.5": 512,
    "SGPT-125M-weightedmean-nli-bitfit": 768,
    "text2vec-large-chinese": 1024,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}

DEFAULT_HF_EMBED_MODEL = "bge-small-zh-v1.5"

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
