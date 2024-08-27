"""Prompts."""

from llama_index.core import PromptTemplate

DEFAULT_QUESTION_GENERATION_PROMPT = '''\
    #01 你是一个问答对数据集处理专家。
    #02 你的任务是根据我给出的内容，生成适合作为问答对数据集的问题。
    #03 问题要关于文件内容，不要太长。
    #04 一句话中只有一个问题。
    #05 生成问题需要具体明确，示例：

    """

    抬高床头有哪些对人体有益的影响？

    舒服德睡眠床如何常驻缩短入睡时间？

    """
    #06 生成问题需要避免指代不明确，以下是需要避免的示例：这款产品、这些文献、这项研究等。
    #07 以下是我给出的内容：
    ---------------------
    {context_str}
    ---------------------
    {query_str}
    '''


DEFAULT_TEXT_QA_PROMPT_TMPL = """内容信息如下
    ---------------------
    {context_str}
    ---------------------
    根据提供内容而非其他知识回答问题.
    问题: {query_str}
    答案: """

DEFAULT_QUESTION_GENERATION_QUERY = "你是一个问答对数据集处理专家。你的任务是产出 \
                        {num_questions_per_chunk} 个问题。 \
                        整个文件中的问题本质上应该是多样化的。将问题限制在所提供的上下文信息范围内。"

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
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

CONDENSE_QUESTION_CHAT_ENGINE_PROMPT_ZH = PromptTemplate(
    """\
给定一次对话（人类和助理之间）以及来自人类的后续消息，\
将消息重写为一个独立的问题，捕获对话中的所有相关上下文。

<聊天记录>
{chat_history}

<后续消息>
{question}

<独立问题>
"""
)


QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a single input query. "
    "Generate {num_queries} search queries in Chinese, one on each line, related to the following input query:\n"
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
    "请根据给定的材料回答给出的问题，如果材料中没有找到答案，就说没有找到相关的信息，不要编造答案。\n\n"
    "如果上面有图片对你生成答案有帮助，请从图片列表中找到对应的图片链接并用markdown格式给出，如![](image_url)。\n\n"
    "---------------------\n\n"
    "问题: {query_str} 。返回文字和图片。\n"
    "答案: "
)
