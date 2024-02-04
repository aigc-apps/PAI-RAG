# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

import re

class CustomPrompt:
    def __init__(self, args):
        self.prompt_type = args.prompt_engineering
        # self.prompt_type = prompt_type

    def simple_prompts(self, contents, question):
        context_docs = ""
        for idx, item in enumerate(contents):
            doc = item[0] if isinstance(item, tuple) else item
            context_docs += f"{str(idx+1)}.{doc.page_content}\n\n"

        prompt_template = "参考内容如下：\n{context}\n作为个人知识答疑助手，请根据上述参考内容回答下面问题，答案中不允许包含编造内容。\n用户问题:\n{question}"
        query_prompt = prompt_template.format(context=context_docs, question=question)

        return query_prompt

    def general_prompts(self, contents, question):
        context_docs = ""
        for idx, item in enumerate(contents):
            doc = item[0] if isinstance(item, tuple) else item
            context_docs += "-----\n\n"+str(idx+1)+".\n"+doc.page_content
        context_docs += "\n\n-----\n\n"

        prompt_template = "基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 \"根据已知信息无法回答该问题\" 或 \"没有提供足够的相关信息\"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context}\n=====\n用户问题:\n{question}"
        query_prompt = prompt_template.format(context=context_docs, question=question)

        return query_prompt

    def extract_url(self, contents, question):
        prompt = '你是一位智能小助手，请根据下面我所提供的相关知识，对我提出的问题进行回答。回答的内容必须包括其定义、特征、应用领域以及相关网页链接等等内容，同时务必满足下方所提的要求！\n 相关知识如下：\n'
        for i, item in enumerate(contents):
            doc = item[0] if isinstance(item, tuple) else item
            doc_page = doc.page_content
            if 'http' in doc_page:
                prompt += str(i + 1) + '、该知识中包含网页链接!' + '\n' + doc_page +'。'+ '\n' + '知识中包含的链接如下:'
                pattern = r"([^：]+)：(https?://\S+?)(?=\s|$)"
                matches = re.findall(pattern, doc_page)
                # 将链接和对应名称内容存储到列表中
                links = [(name.strip(), url) for name, url in matches]
                for name, url in links:
                    prompt +=  '\n' + name + ':' + url + '；'+'\n'
            else:
                prompt += str(i + 1) + '、' + doc_page + '\n'
        if 'http' in prompt:
            requirement =  '回答的内容要求:若提供的知识中存在“网页链接”，则必须将“网页链接”准确无误的输出。不需要输出知识库以外的网页链接'
            prompt += '\n' + requirement + '\n' + '\n' +'问题是：1.' + question + '？' + '2. 上方提供的知识中可供参考的链接有什么' + '？\n'
        else:
            prompt += '\n' +'问题是：' + question + '\n'

        return prompt

    def accurate_content(self, contents, question):
        prompt = '你是一位知识小助手，请根据下面我提供的知识库中相关知识，对我提出的若干问题进行回答，同时回答的内容需满足我所提的要求!\n 知识库相关知识如下：\n'
        for i, item in enumerate(contents):
            doc = item[0] if isinstance(item, tuple) else item
            if 'http' in doc.page_content:
                prompt += str(i + 1) + '、' + doc.page_content +'。'+ '\n'+'以上知识中包含网页链接!'+ '\n'
            elif '超链接' in doc.page_content:
                prompt += str(i + 1) + '、' + doc.page_content +'。'+ '\n'+'以上知识中包含超链接!'+ '\n'

            else:
                prompt += str(i + 1) + '、' + doc.page_content + '\n'
        if 'http' in prompt:
            requirement =  '\n' + '好的，知识库的知识已经提供完毕。同时，我要求你的回答满足以下要求如下几点:'+ '\n'+ '1.知识库中存在“网页链接或超链接，则必须将“网页链接”或“超链接”准确无误的输出，若存在超链接，却不输出超链接，则视为故意隐瞒信息。2.知识库中对于网页链接或超链接前的“文字描述内容”，请准确无误的输出内容。请切记，不允许在回答中添加编造成分。3. 请确保知识库中的网页链接和知识中对网页链接的描述准确无误写出，请不要用修改知识库中针对网页链接或超链接前的文字内容！'
            prompt += '\n' + requirement + '\n' + '\n' +'请根据上方所提供的知识库内容与要求，逐一回答以下几个问题:'+ '\n' +'1. ' + question + '？' +'\n' + '2. 上方知识库中可供参考的“网页链接”有什么？'+ '\n' +'3. 知识库中提供的网页链接前的原文是什么？'+ '\n' +\
            '4. 上方知识库中可供参考的“超链接”后的文字什么？' + '\n' +'5. 你可以确保你提供的每一个网页链接与网页链接的描述，与上方知识库中涉及的网页链接与内容完全一样，无任何自主修改与添加的内容吗？' + '\n' +'你的回答都满足上方提出的要求吗？'
        else:
            prompt += '\n'  +'请根据上方所提供的知识库内容与要求，回答以下问题:' + '\n' + question + '\n'

        return prompt

    def custom_prompts(self, contents, question, prompt):
        context_docs = ""
        for idx, item in enumerate(contents):
            doc = item[0] if isinstance(item, tuple) else item
            context_docs += "-----\n\n"+str(idx+1)+".\n"+doc.page_content
        context_docs += "\n\n-----\n\n"

        query_prompt = prompt.format(context=context_docs, question=question)

        return query_prompt


    def get_prompt(self, docs, query, prompt=None):
        if self.prompt_type == 'customize' and prompt is not None and prompt != '':
            return self.custom_prompts(docs, query, prompt)
        else:
            if self.prompt_type == "simple":
                return self.simple_prompts(docs, query)
            elif self.prompt_type == 'general':
                return self.general_prompts(docs, query)
            elif self.prompt_type == 'extract_url':
                return self.extract_url(docs, query)
            elif self.prompt_type == 'accurate_content':
                return self.accurate_content(docs, query)
            else:
                raise ValueError(f'error: invalid prompt template type of {self.prompt_type}')
