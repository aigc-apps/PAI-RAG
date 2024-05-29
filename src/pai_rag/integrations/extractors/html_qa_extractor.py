from typing import Optional, Dict, Sequence, List, Any
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs
import re
import logging

CHINESE_PUNKTUATION = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。"

logger = logging.getLogger(__name__)


class HtmlQAExtractor(BaseExtractor):
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
        return "HtmlQAExtractor"

    def _deal_Q(self, question, theme, hn, answer, history_QA_dict):
        if hn not in question:
            Q_save = hn + " " + question
        else:
            Q_save = question
        if theme not in question:
            Q_save = theme + " " + Q_save
        else:
            Q_save = Q_save
        if Q_save not in history_QA_dict or len(history_QA_dict[Q_save]) < len(answer):
            history_QA_dict[Q_save] = answer
            return True
        return False

    def _check_answer(self, answer):
        ban_words = ("是什么", "正文中没有", "没有在正文部分提及", "访问错误中心", "参考相关文档", "抱歉", "无法回答")
        for bw in ban_words:
            if bw in answer:
                return False
        return True

    def _check_question(self, question):
        ban_words = ("<h3>",)
        for bw in ban_words:
            if bw in question:
                return False
        return True

    def _filter_html_tags(self, text):
        if not text:
            return text

        filtered_text = re.sub("<code>.*</code>", "", text)
        pattern = "<[^>]{1,999}>"
        html_tag_pattern = re.compile(pattern)
        filtered_text = re.sub(html_tag_pattern, "", filtered_text)
        return filtered_text.strip()

    def _replace_html_tags(self, text):
        pattern = "<[^>]{1,999}>"
        html_tag_pattern = re.compile(pattern)
        new_text = re.sub(html_tag_pattern, "\n", text)
        return re.sub("[\n| |\t]+", "\n", new_text)

    async def _aextract_qa_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract from html nodes only"""
        metadata = {}

        if node.metadata.get("file_type", "Unknown") != "HTML":
            return metadata

        header = node.metadata.get("header", "")

        theme = self._filter_html_tags(header)
        theme = re.sub("^.*?[:|：]", "", theme)

        context_str = node.get_content()
        if not context_str:
            return metadata

        qa_result = {}
        prompt_template = self._get_prompt_template(context_str)
        qa_text = await self.llm.apredict(
            PromptTemplate(template=prompt_template), context_str=context_str
        )
        qa_dict = self._extract_qa_dict(qa_text)

        hn = ""
        q_cnt = 0
        for question, answer in qa_dict.items():
            if not self._check_question(question) or not self._check_answer(answer):
                continue

            if self._deal_Q(question, theme, hn, answer, qa_result):
                q_cnt += 1

        logger.info(f"Generated question count {q_cnt}.")
        metadata["qa_extraction_result"] = qa_result
        metadata["raw_text"] = self._replace_html_tags(context_str)
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
        return QA_dict

    def _get_prompt_template(self, text):
        """
        根据html源码中的关键词，制定与之相应的约束规则，构建QA提取prompt

        Args:
            text (str): html源码

        Returns:
            prompt_prefix (str): 约束规则，每行一条规则
            prompt_tail (str): 空字符串
        """
        _, title_start = re.search("<h1>", text).span()
        title_end, _ = re.search("</h1>", text).span()
        title = text[title_start:title_end]
        # title中不包含空格 中文字符 中文符号时
        # 该html视为对接口/函数的介绍
        # 需要设置问题"%s的功能是什么？" % title
        is_func = True
        zhong_singal = set([c for c in CHINESE_PUNKTUATION])
        for c in title:
            if c == " ":
                is_func = False
                break
            elif c in zhong_singal:
                is_func = False
                break
            elif "\u4e00" <= c <= "\u9fff":
                is_func = False
                break

        prompt_prefix = ""
        keywords = [
            "返回值说明",
            "参数说明",
            "属性列表",
            "命令格式",
            "语法格式",
            "步骤",
            "操作流程",
            "需求分析",
            "修复方案",
            "修复流程",
            "方式1",
            "方式一" "请求参数",
            "返回数据",
            "示例",
            "错误码",
            "返回信息",
            "返回结果",
            "正常返回示例",
            "请求示例",
            "案例",
            "<label>Python",
            "<label>Java",
            "<label>Scala",
        ]
        add_flag = {kw: 1 for kw in keywords}
        # <h1>后<h2>前的前置描述 通常描述了函数/接口功能
        if (
            is_func
            and re.search("<[^>]*h1[^>]*>", text)
            and not re.search("<[^>]*h2[^>]*>", text)
        ):
            prompt_prefix += "提问“%s的功能是什么？”" % title
        if "<table>" in text:
            prompt_prefix += "以<table>作为表格部分，为表格中每一行的内容设置问题\n"
            prompt_prefix += "用制表符”|“和”-“描述<table>结构\n"
        if "<code>" in text:
            prompt_prefix += "整个<code>代码部分作为问题的答案，不分析代码含义及功能，为每个代码部分都设置一个问题\n"
        for kw in keywords:
            if kw in text:
                if add_flag[kw] <= 0:
                    continue
                if kw == "调试":
                    prompt_prefix += "设计一个问题，提问“%s 如何%s？”\n" % (title, kw)
                elif kw == "参数说明":
                    prompt_prefix += "设计一个问题，令这个问题的答案涵盖所有参数的说明信息\n"
                elif kw == "属性列表":
                    prompt_prefix += "设计一个问题，令这个问题的答案涵盖所有属性的说明信息\n"
                elif kw == "方式1" or kw == "方式一":
                    add_flag["方式1"] = 0
                    add_flag["方式一"] = 0
                    prompt_prefix += "设计一个问题，令这个问题的答案涵盖所有”方式”的内容\n"
                elif kw == "示例":
                    prompt_prefix += "为每一个示例设置一个问题，以“进行某某操作的示例是什么？”作为提问格式，以代码作为答案\n"
                elif kw == "案例":
                    prompt_prefix += "额外地为每一个案例设置一个问题，以“进行某某操作的案例是什么？”作为提问格式，以代码作为答案\n"
                elif kw == "<label>Python":
                    prompt_prefix += (
                        "额外地为每一个Python代码设置一个问题，以“进行某某操作的Python代码示例是什么？”作为提问格式，以代码作为答案\n"
                    )
                elif kw == "<label>Java":
                    prompt_prefix += (
                        "额外地为每一个Java代码设置一个问题，以“进行某某操作的Java代码示例是什么？”作为提问格式，以代码作为答案\n"
                    )
                elif kw == "<label>Scala":
                    prompt_prefix += (
                        "额外地为每一个Scala代码设置一个问题，以“进行某某操作的Scala代码示例是什么？”作为提问格式，以代码作为答案\n"
                    )
                # elif kw == "<h3>":
                #     prompt_prefix += "设计问题，提问<h3>下的内容\n"
                elif kw == "步骤":
                    add_flag["操作流程"] = 0
                    prompt_prefix += "提问操作步骤，答案中需要包含所有操作步骤的详细信息，将正文内的形似“1.”的编号后内容理解为步骤的操作顺序，问题中写明操作步骤的主体\n"
                elif kw == "操作流程":
                    add_flag["步骤"] = 0
                    prompt_prefix += "提问操作步骤，答案中需要包含所有操作步骤的详细信息，将正文内的形似“1.”的编号后内容理解为步骤的操作顺序，问题中写明操作步骤的主体\n"
                elif kw == "修复方案":
                    prompt_prefix += "提问“%s应采用什么%s”\n" % (title, kw)
                else:
                    prompt_prefix += "设计一个问题，提问%s，该问题中写明%s的主体\n" % (kw, kw)
        if len(prompt_prefix) > 0:
            prompt_prefix = "以下是对生成问题和答案的要求：\n使用中文生成问题和答案\n" + prompt_prefix

        prompt_template = (
            "从正文部分中提取问题和对应的答案\n"
            + prompt_prefix
            + "回复格式为：\n问题1：xxx\n答案1：xxx\n\n以下是正文部分：\n {context_str}\n"
        )

        return prompt_template
