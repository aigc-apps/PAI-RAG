# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

import re
from typing import List
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, cfg):
        chunk_size = int(cfg['create_docs']['chunk_size'])
        chunk_overlap = cfg['create_docs']['chunk_overlap']
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, docs):
        return self.text_splitter.split_documents(docs)

    def split_text1(self, text: str, isPDF: bool = False) -> List[str]:
        if isPDF:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


    def split_text2(self, text: str, sentence_size: int, isPDF: bool = False) -> List[str]:
        if isPDF:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        if not isinstance(text, str):
            raise ValueError('Input must be str type.')

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)  # 特殊处理引号前的断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)

        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.replace(" ", "")
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > sentence_size:
                # print("----长度过长----超700")
                ele1 = re.sub(r'([。.]["’”」』]{0,2})([^,，.。])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                ele1_ls += [''] * (8 - len(ele1_ls) % 8) #将列表长度补足为3的倍数
                ele1_ls = [ele1_ls[i] + ele1_ls[i+1]+ ele1_ls[i+2]+ ele1_ls[i+3]+ ele1_ls[i+4]+ ele1_ls[i+5]+ ele1_ls[i+6]+ ele1_ls[i+7]for i in range(0, len(ele1_ls), 8)]

                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")


                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                        ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls
