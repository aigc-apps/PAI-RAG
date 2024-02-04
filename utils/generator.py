import time, re
import zhon
import openai
import requests
import json
from modules.CustomLLM import CustomLLM
from langchain.llms import OpenAI
from modules.LocalLLM import LocalLLM

class HtmlGenerator:
    def __init__(self, config) -> None:
        self.config = config
        if self.config['LLM'] == 'EAS':
            self.llm = CustomLLM()
            self.llm.url = self.config['EASCfg']['url']
            self.llm.token = self.config['EASCfg']['token']
            self.llm.top_k = int(30)
            self.llm.top_p = float(0.8)
            self.llm.temperature = float(0.7)
        elif self.config['LLM'] == 'OpenAI':
            self.llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key=self.config['OpenAI']['key'])
        elif self.config['LLM'] == 'Local':
            print(f"[INFO] loading qa extraction model from local: {self.config['local_model_path']}")
            self.llm = LocalLLM(model_name_or_path=self.config['local_model_path'])

    def select_prompt(self, QA_text):
        _, title_start = re.search("<h1>", QA_text).span()
        title_end, _ = re.search("</h1>", QA_text).span()
        title = QA_text[title_start:title_end]
        # title中不包含空格 中文字符 中文符号时
        # 该html视为对接口/函数的介绍
        # 需要设置问题"%s的功能是什么？" % title
        is_func = True
        zhong_singal = set([c for c in zhon.hanzi.punctuation])
        for c in title:
            if c == " ":
                is_func = False
                break
            elif c in zhong_singal:
                is_func = False
                break
            elif '\u4e00' <= c <= '\u9fff':
                is_func = False
                break

        prompt_prefix = ""
        prompt_tail = ""
        keywords = [
            "返回值说明", "参数说明", "属性列表", "命令格式", "语法格式", "步骤", "操作流程", "需求分析", "修复方案", "修复流程", "方式1", "方式一"
            "请求参数", "返回数据", "示例", "错误码", "返回信息", "返回结果", "正常返回示例", "请求示例",
            "案例", "<label>Python", "<label>Java", "<label>Scala"
        ]
        add_flag = {kw:1 for kw in keywords}
        # <h1>后<h2>前的前置描述 通常描述了函数/接口功能
        if is_func and re.search("<[^>]*h1[^>]*>", QA_text) and not re.search("<[^>]*h2[^>]*>", QA_text):
            prompt_prefix += "提问“%s的功能是什么？”" % title
        if "<table>" in QA_text:
            prompt_prefix += "以<table>作为表格部分，为表格中每一行的内容设置问题\n"
            prompt_prefix += "用制表符”|“和”-“描述<table>结构\n"
        if "<code>" in QA_text:
            prompt_prefix += "整个<code>代码部分作为问题的答案，不分析代码含义及功能，为每个代码部分都设置一个问题\n"
        for kw in keywords:
            if kw in QA_text:
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
                    prompt_prefix += "额外地为每一个Python代码设置一个问题，以“进行某某操作的Python代码示例是什么？”作为提问格式，以代码作为答案\n"
                elif kw == "<label>Java":
                    prompt_prefix += "额外地为每一个Java代码设置一个问题，以“进行某某操作的Java代码示例是什么？”作为提问格式，以代码作为答案\n"
                elif kw == "<label>Scala":
                    prompt_prefix += "额外地为每一个Scala代码设置一个问题，以“进行某某操作的Scala代码示例是什么？”作为提问格式，以代码作为答案\n"
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
        return prompt_prefix, prompt_tail

    # 功能：使用指定模型对input_text生成QA
    # 且QA的格式为：问题n：xxx\n答案n：xxx\n\n"(这部分的提示在prompt_out_pratten中体现)
    # 传入：
    #   model_engine:str 选用何种openai模型
    #   input_text:str 需要提取QA的目标文本
    #   prompt_prefix:str 前置prompt
    #   prompt_tail:str 后置prompt
    #   try_lim:int 考虑到openai无法正常调用是由于多种情况导致的，限定尝试次数，达到尝试次数后产生报错信息终止该函数
    #      报错内容：RuntimeError("[Openai Error]在OPENAI的调用部分保持 大概率是text过长导致的错误")
    def get_QA_for_text(
            self,
            input_text,
            prompt_prefix,
            prompt_tail
    ):
        prompt_task = "从正文部分中提取问题和对应的答案\n"
        prompt_out_pratten = "回复格式为：\n问题1：xxx\n答案1：xxx\n\n以下是正文部分：\n"
        input_text = prompt_task + prompt_prefix + prompt_out_pratten + input_text + prompt_tail

        try_lim = 10
        try_cnt = 0
        while try_cnt <= try_lim:
            try:
                collected_message = self.llm(input_text)
                print(f"[INFO] LLM Response:\n{collected_message}")
                return collected_message
            except Exception as e:
                error_message = str(e)
                if try_cnt >= try_lim:
                    raise RuntimeError("[Openai Error]在OPENAI的调用部分保持 大概率是text过长导致的错误 error_message:%s" % error_message)
                try_cnt += 1

    # 功能：将”问题1：xxx\n答案1：xxx\n“格式的QA文本处理为键值对 {Q:A}
    #   在处理过程中需要过滤掉Q中的html标签
    #   过滤掉A中除<table> <tr> <td>等制表符之外的标签
    # 传入：
    #   QA_text:str 格式化的QA文本
    # 传出：
    #   QA_dict:dict {Q:A}
    def QAtext2QAdict(self, QA_text):
        if len(QA_text) == 0:
            return None
        partten_question = re.compile("问题[0-9]+[：:]")
        partten_answer = re.compile("答案[0-9]+[:：]|回答[0-9]+[:：]")
        time.sleep(0.002)
        Q_index = [obj.span() for obj in list(partten_question.finditer(QA_text))]
        A_index = [obj.span() for obj in list(partten_answer.finditer(QA_text))]
        if len(Q_index) != len(A_index) or len(Q_index) == 0 or len(A_index) == 0:
            print("text: ", QA_text)
            print("Q_index: ", Q_index)
            print("A_index: ", A_index)
            raise IndexError("[To Dict Error]提取出的问题和答案的数量不一致")
        QA_i = 0
        QA_list = [[], []]
        while QA_i < len(Q_index):
            QA_list[0].append(QA_text[Q_index[QA_i][1]:A_index[QA_i][0]].strip())
            if QA_i + 1 < len(Q_index):
                QA_list[1].append(QA_text[A_index[QA_i][1]:Q_index[QA_i+1][0]].strip())
            QA_i += 1
        # 处理最后一个A
        QA_list[1].append(QA_text[A_index[-1][1]:].strip())
        QA_dict = {QA_list[0][i]:QA_list[1][i] for i in range(len(Q_index))}
        # {Q1: A1, Q2: A2}
        return QA_dict

    def generateQA(
        self,
        src_text
    ):
        prompt_prefix, prompt_tail = self.select_prompt(src_text)

        summary_dict = {}

        print("[INFO] Extracting QA from sub doc...")
        try_lim = 10
        try_cnt = 0
        while try_cnt <= try_lim:
            try:
                QA_text = self.get_QA_for_text(src_text, prompt_prefix, prompt_tail)
                QA_dict = self.QAtext2QAdict(QA_text)
                if QA_dict:
                    summary_dict.update(QA_dict)
                break
            except IndexError as e:
                print(str(e))
                if try_cnt <= try_lim:
                    try_cnt += 1
                else:
                    raise RuntimeError("[To Dict Error]提取出的问题和答案的数量不一致")
        return summary_dict