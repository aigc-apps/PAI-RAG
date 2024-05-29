from typing import Optional, Dict, Sequence, List, Any
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs
import os
import re
import logging

logger = logging.getLogger(__name__)


class TextQAExtractor(BaseExtractor):
    llm: Optional[LLM] = Field(description="The LLM to use for generation.")

    def __init__(
        self,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ):
        super().__init__(
            llm=llm,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextQAExtractor"

    async def _aextract_qa_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract from html nodes only"""
        metadata = {}
        file_path = node.metadata.get("file_path", "")
        file_extension = os.path.splitext(file_path)[1]
        if file_extension != ".txt":
            return metadata

        context_str = node.get_content()
        if not context_str:
            return metadata

        qa_result = {}
        prompt_template = self._get_prompt_template()
        qa_text = await self.llm.apredict(
            PromptTemplate(template=prompt_template), context_str=context_str
        )
        qa_dict = self._extract_qa_dict(qa_text)

        q_cnt = 0
        for question, answer in qa_dict.items():
            if not self._check_question(question) or not self._check_answer(answer):
                continue
            qa_result[question] = answer
            q_cnt += 1

        logger.info(f"Generated question count {q_cnt}.")
        metadata["qa_extraction_result"] = qa_result
        return metadata

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        questions_jobs = []
        for node in nodes:
            questions_jobs.append(self._aextract_qa_from_node(node))

        metadata_list: List[Dict] = await run_jobs(
            questions_jobs, show_progress=self.show_progress, workers=self.num_workers
        )

        return metadata_list

    # 功能：将”问题1：xxx\n答案1：xxx\n“格式的QA文本处理为键值对 {Q:A}
    #   在处理过程中需要过滤掉Q中的html标签
    #   过滤掉A中除<table> <tr> <td>等制表符之外的标签
    # 传入：
    #   text:str 格式化的QA文本
    # 传出：
    #   qa_dict:dict {Q:A}
    def _extract_qa_dict(self, text):
        """
        从qa提取llm的回复中提取出qa对

        Args:
            text (str): qa提取llm的回复结果

        Returns:
            qa_dict (dict): qa对字典
            {Q1: A1, Q2: A2}
        """
        if len(text) == 0:
            return None
        partten_question = re.compile("问题[0-9]+[：:]")
        partten_answer = re.compile("答案[0-9]+[:：]|回答[0-9]+[:：]")
        Q_index = [obj.span() for obj in list(partten_question.finditer(text))]
        A_index = [obj.span() for obj in list(partten_answer.finditer(text))]
        if len(Q_index) != len(A_index) or len(Q_index) == 0 or len(A_index) == 0:
            print("text: ", text)
            print("Q_index: ", Q_index)
            print("A_index: ", A_index)
            if len(Q_index) == len(A_index) + 1:
                # 截断的情况
                Q_index = Q_index[:-1]
            else:
                raise IndexError("[To Dict Error]提取出的问题和答案的数量不一致")
        QA_i = 0
        QA_list = [[], []]
        while QA_i < len(Q_index):
            QA_list[0].append(text[Q_index[QA_i][1] : A_index[QA_i][0]].strip())
            if QA_i + 1 < len(Q_index):
                QA_list[1].append(text[A_index[QA_i][1] : Q_index[QA_i + 1][0]].strip())
            QA_i += 1
        # 处理最后一个A
        QA_list[1].append(text[A_index[-1][1] :].strip())
        QA_dict = {QA_list[0][i]: QA_list[1][i] for i in range(len(Q_index))}
        # {Q1: A1, Q2: A2}
        print(QA_dict)
        return QA_dict

    def _get_prompt_template(self):
        """
        根据原始文档中的关键词，制定与之相应的约束规则，构建QA提取prompt

        Args:
            QA_text (str): 原始文档

        Returns:
            prompt (str)
        """
        return """
            从正文部分中提取问题和对应的答案。
            使用中文生成问题和答案。
            回复格式为：
            问题1：xxx
            答案1：xxx

            以下是正文部分:
            {context_str}"""

    def _check_answer(self, answer):
        ban_words = (
            "是什么",
            "正文中没有",
            "没有在正文部分提及",
            "访问错误中心",
            "参考相关文档",
            "抱歉",
            "无法回答",
            "谢谢",
            "不客气",
        )
        for bw in ban_words:
            if bw in answer:
                return False
        return True

    def _check_question(self, question):
        ban_words = ("<h3>", "谢谢", "不客气")
        for bw in ban_words:
            if bw in question:
                return False
        return True
