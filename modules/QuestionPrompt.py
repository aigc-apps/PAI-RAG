from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

_template_en = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT_EN = PromptTemplate.from_template(_template_en)

_template_ch = """请根据聊天记录和新问题，将新问题改写为一个独立问题。
不需要回答问题，一定要返回一个疑问句。
聊天记录：
{chat_history}
新问题：{question}
独立问题："""
CONDENSE_QUESTION_PROMPT_CH = PromptTemplate.from_template(_template_ch)

def get_standalone_question_en(llm):
    question_generator_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT_EN)
    return question_generator_chain

def get_standalone_question_ch(llm):
    question_generator_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT_CH)
    return question_generator_chain
