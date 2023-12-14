# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

import json
import time
import os
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from .CustomPrompt import CustomPrompt
from .EASAgent import EASAgent
from .VectorDB import VectorDB
from .TextSplitter import TextSplitter
import nltk
from .CustomLLM import CustomLLM
from .QuestionPrompt import *
from sentencepiece import SentencePieceProcessor
from langchain.llms import OpenAI
import argparse
import json
from typing import Iterable, List
from .EasLlmClient import EasLlmClient

import requests

class LLMService:
    def __init__(self, args):
        # assert args.upload or args.user_query, "error: dose not set any action, please set '--upload' or '--query <user_query>'."
        # assert os.path.exists(args.config), f"error: config path {args.config} does not exist."
        self.langchain_chat_history = []
        self.input_tokens = []
        self.llm_chat_history = []
        self.sp = SentencePieceProcessor(model_file='./tokenizer.model')
        self.topk = 3
        self.prompt_type = 'general'
        self.prompt =  "基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 \"根据已知信息无法回答该问题\" 或 \"没有提供足够的相关信息\"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context}\n=====\n用户问题:\n{question}"
        nltk_data_path = "/code/nltk_data"
        if os.path.exists(nltk_data_path):
            nltk.data.path = [nltk_data_path] + nltk.data.path
            
        # with open(args.config) as f:
        #     cfg = json.load(f)
        # self.init_with_cfg(cfg, args)

    def init_with_cfg(self, cfg, args):
        self.cfg = cfg
        self.args = args

        # self.prompt_template = PromptTemplate(self.args)
        # self.eas_agent = EASAgent(self.cfg)
        self.vector_db = VectorDB(self.args, self.cfg)
        
        print('self.cfg ', self.cfg)
        self.llm = None
        if self.cfg['LLM'] == 'EAS':
            self.llm_ = CustomLLM()
            self.llm_.url = self.cfg['EASCfg']['url']
            self.llm_.token = self.cfg['EASCfg']['token']
            
            self.llm = EasLlmClient(host=self.cfg['EASCfg']['url'], authorization=self.cfg['EASCfg']['token'])
        elif self.cfg['LLM'] == 'OpenAI':
            self.llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key = self.cfg['OpenAI']['key'])
        self.question_generator_chain = get_standalone_question_ch(self.llm_)

    def upload_custom_knowledge(self, docs_dir=None, chunk_size=200,chunk_overlap=0):
        if docs_dir is None:
            docs_dir = self.cfg['create_docs']['docs_dir']
        self.cfg['create_docs']['chunk_size'] = chunk_size
        self.cfg['create_docs']['chunk_overlap'] = chunk_overlap
        self.text_splitter = TextSplitter(self.cfg)
        if os.path.isdir(docs_dir):
            docs = DirectoryLoader(docs_dir, glob=self.cfg['create_docs']['glob'], show_progress=True).load()
            docs = self.text_splitter.split_documents(docs)
        else:
            loader = UnstructuredFileLoader(docs_dir, mode="elements")
            docs = loader.load_and_split(text_splitter=self.text_splitter)

        start_time = time.time()
        print('Uploading custom knowledge.', start_time)
        self.vector_db.add_documents(docs)
        end_time = time.time()
        print("Insert Success. Cost time: {} s".format(end_time - start_time))

    def create_user_query_prompt(self, query, topk, prompt_type, prompt=None, score_threshold=0.5):
        if topk == '' or topk is None:
            topk = 3
        docs = self.vector_db.similarity_search_db(query, topk=int(topk),score_threshold=float(score_threshold))
        print('create_user_query_prompt docs', docs)
        if prompt_type == "General":
            self.args.prompt_engineering = 'general'
        elif prompt_type == "Extract URL":
            self.args.prompt_engineering = 'extract_url'
        elif prompt_type == "Accurate Content":
            self.args.prompt_engineering = 'accurate_content'
        elif prompt_type == "Customize":
            self.args.prompt_engineering = 'customize'
        self.prompt_template = CustomPrompt(self.args)
        user_prompt = self.prompt_template.get_prompt(docs, query, prompt)

        return user_prompt 

    def get_new_question(self, query):
        if len(self.langchain_chat_history) == 0:
            print('result',query)
            return query
        else:
            result = self.question_generator_chain({"question": query, "chat_history": self.langchain_chat_history})
            print('result',result)
            return result['text']

    def checkout_history_and_summary(self, summary=False,llm_topK=30, llm_topp=0.8, llm_temp=0.7):
        if summary or len(self.langchain_chat_history) > 10:
            print("start summary")
            if self.cfg['LLM'] == 'EAS':
                llm_history = self.langchain_chat_history
                response = self.llm.post_http_request(prompt="请对我们之前的对话内容进行总结。",system_prompt="",history=llm_history,temperature=llm_temp,top_k=llm_topK,top_p=llm_topp,use_stream_chat=False)
                summary_res, _ = self.llm.get_response(response)
            elif self.cfg['LLM'] == 'OpenAI':
                summary_res = self.llm(f"question: 请对我们之前的对话内容进行总结。 chat_history: {self.langchain_chat_history}")
            print("请对我们之前的对话内容进行总结: ", summary_res)
            self.langchain_chat_history = []
            self.langchain_chat_history.append(("请对我们之前的对话内容进行总结。", summary_res))
            self.input_tokens = []
            self.input_tokens.append("请对我们之前的对话内容进行总结。")
            self.input_tokens.append(summary_res)
            return summary_res
        else:
            return ""
    
    
    def get_streaming_response(self, response) -> Iterable[List[str]]:
        delimiter = b'\0'  # selected in [b'\n', b'\0']
        print(f"response: {response}")
        yield response


    def get_streaming_response(self, response) -> Iterable[List[str]]:
        delimiter = b'\0'  # selected in [b'\n', b'\0']
        print(f"response: {response}")
        yield response

    def query_retrieval_llm(self, query, topk='', score_threshold=0.5, prompt_type='', prompt=None, history=False, llm_topK=30, llm_topp=0.8, llm_temp=0.7):
        
        new_query=self.get_new_question(query) if (history) else query
        print('new_query', new_query)
        topk = self.topk if (topk == '') else topk
        prompt_type = self.prompt_type if (prompt_type == '') else prompt_type
        prompt = self.prompt if( prompt is None) else prompt
    
        user_prompt = self.create_user_query_prompt(new_query, topk, prompt_type, prompt, score_threshold)
        print(f"Post user query to {self.cfg['LLM']}")
        if self.cfg['LLM'] == 'EAS':
            if history:
                llm_history = self.langchain_chat_history
            else:
                llm_history = []                
            print(f"LLM-EAS: query: {user_prompt}, history: {llm_history}, top_k:{llm_topK}, top_p:{llm_topp}, temperature:{llm_temp}")
            response = self.llm.post_http_request(prompt=user_prompt,system_prompt="",history=llm_history,temperature=llm_temp,top_k=llm_topK,top_p=llm_topp,use_stream_chat=False)
            ans, _ = self.llm.get_response(response)
            
        elif self.cfg['LLM'] == 'OpenAI':
            llm_topp = float(llm_topp) if llm_topp is not None else 1.0
            llm_temp = float(llm_temp) if llm_temp is not None else 0.7
            self.llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key = self.cfg['OpenAI']['key'], temperature=llm_temp, top_p=llm_topp)
            if history:
                print(f"LLM-OpenAI: query: {user_prompt}, history: {self.langchain_chat_history}, top_p:{llm_topp}, temperature:{llm_temp}")
                ans = self.llm(f"question: {user_prompt}, chat_history: {self.langchain_chat_history}")
            else:
                print(f"LLM-OpenAI: query: {user_prompt}, history: [], top_p:{llm_topp}, temperature:{llm_temp}")
                ans = self.llm(query)
        if history:
            self.langchain_chat_history.append((new_query, ans))
        print(f"Get response from {self.cfg['LLM']}")
        self.input_tokens.append(new_query)
        self.input_tokens.append(ans)
        tokens_len = self.sp.encode(self.input_tokens, out_type=str)
        lens = sum(len(tl) for tl in tokens_len)
        summary_res = self.checkout_history_and_summary(False,llm_temp,llm_topK,llm_topp)
        return ans, lens, summary_res

    def query_only_llm(self, query, history=False, llm_topK=30, llm_topp=0.8, llm_temp=0.7):
        print(f"Post user query to {self.cfg['LLM']}")
        start_time = time.time()
        if self.cfg['LLM'] == 'EAS':
            if history:
                llm_history = self.langchain_chat_history
            else:
                llm_history = []
            
            print(f"LLM-EAS:  query: {query}, history: {llm_history}, top_k:{llm_topK}, top_p:{llm_topp}, temperature:{llm_temp}")
            response = self.llm.post_http_request(prompt=query,system_prompt="",history=llm_history,temperature=llm_temp,top_k=llm_topK,top_p=llm_topp,use_stream_chat=False)
            ans, _ = self.llm.get_response(response)
                        
        elif self.cfg['LLM'] == 'OpenAI':
            llm_topp = float(llm_topp) if llm_topp is not None else 1.0
            llm_temp = float(llm_temp) if llm_temp is not None else 0.7

            self.llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key = self.cfg['OpenAI']['key'], temperature=llm_temp, top_p=llm_topp)
            if history:
                print(f"LLM-OpenAI:  vquestion: {query}, chat_history: {self.langchain_chat_history}, top_p:{llm_topp}, temperature:{llm_temp}")
                ans = self.llm(f"question: {query}, chat_history: {self.langchain_chat_history}")
            else:
                print(f"LLM-OpenAI: question: {query}, history: [], top_p:{llm_topp}, temperature:{llm_temp}")
                ans = self.llm(f"question: {query}")
        if history:
            self.langchain_chat_history.append((query, ans))
        end_time = time.time()
        print(f"Get response from {self.cfg['LLM']}. Cost time: {end_time - start_time} s")
        self.input_tokens.append(query)
        self.input_tokens.append(ans)
        tokens_len = self.sp.encode(self.input_tokens, out_type=str)
        lens = sum(len(tl) for tl in tokens_len)
        summary_res = self.checkout_history_and_summary(False,llm_temp,llm_topK,llm_topp)
        return ans, lens, summary_res
    
    def query_only_vectorstore(self, query, topk='',score_threshold=0.5):
        print("Post user query to Vectore Store")
        if topk is None:
            topk = 3
        
        if topk == '':
            topk = self.topk
        else:
            self.topk = topk
            
        start_time = time.time()
        print('query',query)
        docs = self.vector_db.similarity_search_db(query, topk=int(topk),score_threshold=float(score_threshold))
        print('docs', docs)
        page_contents, ref_names = [], []

        for idx, doc in enumerate(docs):
            content = doc[0].page_content if hasattr(doc[0], "page_content") else "[Doc Content Lost]"
            page_contents.append('='*20 + f' Doc [{idx+1}] ' + '='*20 + f'\n{content}\n')
            ref = doc[0].metadata['filename'] if hasattr(doc[0], "metadata") and "filename" in doc[0].metadata else "[Doc Name Lost]"
            ref_names.append(f'[{idx+1}] {ref}  |  Relevance score: {doc[1]}')

        ref_title = '='*20 + ' Reference Sources ' + '='*20
        context_docs = '\n'.join(page_contents) + f'{ref_title}\n' + '\n'.join(ref_names)
        if len(docs) == 0:
            context_docs = f"No relevant docs were retrieved using the relevance score {score_threshold}."
        end_time = time.time()
        print("Get response from Vectore Store. Cost time: {} s".format(end_time - start_time))
        tokens_len = self.sp.encode(context_docs, out_type=str)
        lens = sum(len(tl) for tl in tokens_len)
        return context_docs, lens
