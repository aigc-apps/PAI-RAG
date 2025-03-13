"""Prompts."""

from llama_index.core import PromptTemplate

DEFAULT_QUESTION_GENERATION_PROMPT = """\
    #01 你是一个问答对数据集处理专家。
    #02 你的任务是根据我给出的内容，生成适合作为问答对数据集的问题。
    #03 问题要关于文件内容，不要太长。
    #04 一句话中只有一个问题。
    #05 生成问题需要具体明确。
    #06 生成问题需要避免指代不明确，以下是需要避免的示例：这款产品、这些文献、这项研究等。
    #07 以下是我给出的内容：
    ---------------------
    {context_str}
    ---------------------
    #08 请仔细阅读给出的内容，生成适合作为问答对数据集的{num_questions_per_chunk}个问题：
    """

DEFAULT_MULTI_MODAL_QUESTION_GENERATION_PROMPT = """\
    #01 你是一个问答对数据集处理专家，擅长理解和分析多模态信息（文字和图片）。
    #02 你的任务是根据我给出的文字内容和相关图像，生成适合作为问答对数据集的问题。
    #03 问题要紧扣文件内容和图像，确保每个问题都清晰且简短。
    #04 一句话中仅包含一个问题。
    #05 生成的问题需要具体明确，能够准确反映文件内容和图像信息。
    #06 生成问题需要避免指代不明确，以下是需要避免的示例：这款产品、这些文献、这项研究等。
    #07 以下是我给出的文字内容和相关图像链接：
    ---------------------
    {context_str}
    ---------------------
    #08 请仔细阅读给出的内容和图像描述，生成适合作为问答对数据集的{num_questions_per_chunk}个问题：
    """

DEFAULT_TEXT_QA_PROMPT_TMPL = """内容信息如下
    ---------------------
    {context_str}
    ---------------------
    根据提供内容而非其他知识回答问题.
    问题: {query_str}
    答案: """


DEFAULT_QA_GENERATE_PROMPT_TMPL_ZH = """\
上下文信息如下。

---------------------
{context_str}
---------------------

给定上下文信息而不是先验知识。
仅生成基于以下查询的问题。

您是一名教师/教授。 \
您的任务是为即将到来的测验/考试设置 \
    {num_questions_per_chunk} 个问题。
整个文件中的问题本质上应该是多样化的。 \
将问题限制在所提供的上下文信息范围内。"
"""

EVALUATION_PYDANTIC_FORMAT_TMPL = """
Here's a JSON schema to follow:
{schema}

Output a valid JSON object but do not repeat the schema.
The response should be concise to keep json complete。
"""


CONDENSE_QUESTION_CHAT_ENGINE_PROMPT = PromptTemplate(
    """\
Please play the role of an intelligent search rewriting and completion robot.
According to the user's chat history,
please rewrite the new question into a condensed question by resolving references from context.
Note: Do not change the meaning of the new question, the answer should be as concise as possible, do not directly answer the question, and do not output more content.

Please think carefully and give your answer using the same language as the <New question>

Example 1:
<Chat history>
User: What did you do this morning?
Assistant: Go play basketball

<New question>
User: Is it fun?

<Condensed question>
Is playing basketball fun?

Example 2:
<Chat history>
User: 有多少只猫?
Assistant: 有1只猫

<New question>
User: 狗呢

<Condensed question>
有多少条狗?

Now it's your turn:
<Chat history>
{chat_history}

<New question>
{question}

<Condensed question>
"""
)


INTENT_REWRITE_PROMPT_ZH = """
# 角色
你是一位专业的聊天记录分析专家，可以根据对话内容确定用户的意图，生成更加精确的查询。

## 技能

### 技能 1: 新闻热榜互动
- 根据对话内容，判断是否需要提供时事新闻、热点新闻资讯等相关查询。
- 如果用户想要查询热门榜单，请生成一个意图为list_news, 生成结果格式为 JSON 对象：```{ "intent": "list_news" }```。
- 如果用户想要了解科技、娱乐、经济、时政、社会、体育、教育、国际等特定板块的新闻，或者想要查询某个热点新闻，请生成一个意图为chat_news。
- 当意图为chat_news时，你需要根据上下文信息对用户的搜索意图进行查询改写，改写之后的意图和查询格式为 JSON 对象：```{ "intent": "chat_news", "query": "new query" }```。


### 技能 2: 互联网搜索
- 根据对话内容，判断是否需要从互联网搜索信息来完成对话内容，如果需要进行互联网搜索，你会分析聊天记录并生成一条搜索查询。
- 如果用户想要查询容易随着时间变化的信息，请生成一个意图为search_web。
- 生成的搜索查询应简洁、明确、与主题相关，尽可能精准，以便获取更多相关信息。
- 时间相关查询
  - 高频波动信息（如黄金价格、外汇汇率、股票价格等）：请提供具体且最新的时间信息，例如最新一天或实时数据，并使用适当的短时间间隔。如今天为2025年1月1日,搜索"xxxx最新股价"改写为`2025年1月1日xxxx股价`.
  - 低频更新信息（如汽车评测、电影上映、歌曲发布等）：请使用较宽泛的时间范围，如最近一个月或更长时间，并提供相关的时间信息。如今天为2025年1月1日,搜索`最近好看的电影`改写为`2025年1月好看的电影`。
  - 极少更新信息但会随着时间变化信息（如政治信息、时效性事实信息、知识查询等）：请使用最新数据，如现在或者最新一天，并提供相关的时间信息。如今天为2025年1月1日,搜索`阿里巴巴总部在哪`改写为`现在阿里巴巴总部在哪`。
- 非时间相关查询：避免随意添加时间信息，确保回答专注于查询的主要内容。
- 生成的意图和查询格式为 JSON 对象：```{ "intent": "search_web", "query": "new query" }```。


### 技能 3: 无需外部信息的问答
- 如果确定不需要查询上面的信息，请直接生成一个意图为chat的 JSON 对象: ``` { "intent": "chat" } ```


## 限制
- **仅**以 JSON 对象的形式响应，不允许任何形式的额外评论、解释或附加文本。
- 除非绝对确定没有有用的结果可以通过新闻或者互联网搜索获得，否则建议生成新闻或者互联网搜索查询。
- 保持输出格式的一致性，严格遵循给定的 JSON 格式要求。
- 互联网搜索查询生成时，应该简明扼要地专注于撰写高质量的搜索查询，避免不必要的详细说明、评论或假设。
- 除非用户要求，否则保持输出语种与用户输入问题语种的一致性。
"""


CONDENSE_QUESTION_CHAT_ENGINE_PROMPT_ZH = """# 角色
你是一位专业的信息检索专家，负责分析聊天记录以确定是否需要生成搜索查询。你的目标是确保获取全面、最新且有价值的信息。

## 技能
### 技能 1: 聊天记录分析
- 分析提供的聊天记录，判断是否需要生成搜索查询。
- 如果存在任何不确定性或可能获取到有用信息的情况，只需生成1个相关且精确的搜索查询。

### 技能 2: 生成搜索查询
- 生成的搜索查询应简洁、明确且与主题相关。
- 查询应尽可能精准，以便获取更多相关信息。
- 时间相关查询
  - 高频波动信息（如黄金价格、外汇汇率、股票价格等）：请提供具体且最新的时间信息，例如最新一天或实时数据，并使用适当的短时间间隔。如今天为2025年1月1日,搜索"xxxx最新股价"改写为`2025年1月1日xxxx股价`.
  - 低频更新信息（如汽车评测、电影上映、歌曲发布等）：请使用较宽泛的时间范围，如最近一个月或更长时间，并提供相关的时间信息。如今天为2025年1月1日,搜索`最近好看的电影`改写为`2025年1月好看的电影`。
  - 极少更新信息但会随着时间变化信息（如政治信息、时效性事实信息、知识查询等）：请使用最新数据，如现在或者最新一天，并提供相关的时间信息。如今天为2025年1月1日,搜索`阿里巴巴总部在哪`改写为`现在阿里巴巴总部在哪`。
- 非时间相关查询：避免随意添加时间信息，确保回答专注于查询的主要内容。
- 生成的查询格式为 JSON 对象：```{ "query": "new query" }```。

### 技能 3: 确定无需搜索
- 如果完全确定不需要额外信息，返回空字符串：```{ "query": "" }```。

## 限制
- **仅**以 JSON 对象的形式响应，不允许任何形式的额外评论、解释或附加文本。
- 除非绝对确定没有有用的结果可以通过搜索获得，否则建议生成搜索查询。
- 在生成搜索查询时，确保每个查询都是独立的、简洁的，并且与主题相关。
- 保持输出格式的一致性，严格遵循给定的 JSON 格式要求。
- 简明扼要地专注于撰写高质量的搜索查询，避免不必要的详细说明、评论或假设。
- 除非用户要求，否则保持输出语种与用户输入问题语种的一致性。
"""

CONDENSE_QUESTION_ANSWER_PROMPT_ZH = """## 聊天记录:
{chat_history}

用户:
{question}

请仔细思考后，给出你的答案。除非用户要求，否则请保持输出语种与用户输入问题语种的一致性：
"""

QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a single input query. "
    "Generate {num_queries} search queries in Chinese, one on each line, related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)

DEFAULT_FUSION_TRANSFORM_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the provided text in Chinese, including as many key details as needed."
)

DEFAULT_MULTI_MODAL_TEXT_QA_PROMPT_TMPL = (
    "结合上面给出的图片和下面给出的参考材料来回答用户的问题。\n\n"
    "参考材料:"
    "---------------------\n\n"
    "{context_str}\n"
    "---------------------\n\n"
    "请根据给定的材料回答给出的问题，如果材料中没有找到答案，就说没有找到相关的信息，不要编造答案。\n\n"
    "---------------------\n\n"
    "问题: {query_str}\n"
    "答案: "
)

DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL = (
    "结合上面给出的图片和下面给出的参考材料来回答用户的问题。材料中包含一组图片链接，分别对应到前面给出的图片的地址。\n\n"
    "材料:"
    "---------------------\n\n"
    "{context_str}\n"
    "---------------------\n\n"
    "请根据给定的材料回答给出的问题，回答中需要有文字描述和图片。如果材料中没有找到答案，就说没有找到相关的信息，不要编造答案。\n\n"
    "如果上面有图片对你生成答案有帮助，请找到图片链接并用markdown格式给出，如![](image_url)。\n\n"
    "---------------------\n\n"
    "问题: {query_str}\n请返回文字和展示图片，不需要标明图片顺序"
    "答案: "
)
