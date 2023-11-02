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

class LLMService:
    def __init__(self, args):
        # assert args.upload or args.user_query, "error: dose not set any action, please set '--upload' or '--query <user_query>'."
        # assert os.path.exists(args.config), f"error: config path {args.config} does not exist."
        self.langchain_chat_history = []
        self.input_tokens = []
        self.llm_chat_history = []
        self.sp = SentencePieceProcessor(model_file='./tokenizer.model')
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
        
        self.llm = CustomLLM()
        self.llm.url = self.cfg['EASCfg']['url']
        self.llm.token = self.cfg['EASCfg']['token']
        self.question_generator_chain = get_standalone_question_ch(self.llm)

        # if args.upload:
        #     self.upload_custom_knowledge()
        # if args.user_query:
        #     if args.query_type == "retrieval_llm":
        #         self.query_func = self.query_retrieval_llm
        #         self.query_type = "Retrieval-Augmented Generation"
        #     elif args.query_type == "only_llm":
        #         self.query_func = self.query_only_llm
        #         self.query_type = "Vanilla-LLM Generation"
        #     elif args.query_type == "only_vectorstore":
        #         self.query_func = self.query_only_vectorstore
        #         self.query_type = "Vector-Store Retrieval"
        #     else:
        #         raise ValueError(f'error: invalid query type of {args.query_type}')

        #     answer = self.query_func(args.user_query)
        #     print('='*20 + f' {self.query_type} ' + '='*20 + '\n', answer)

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

    def create_user_query_prompt(self, query, topk, prompt_type, prompt=None):
        if topk == '' or topk is None:
            topk = 3
        docs = self.vector_db.similarity_search_db(query, topk=int(topk))
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

    def checkout_history_and_summary(self, summary=False):
        if summary or len(self.langchain_chat_history) > 10:
            print("start summary")
            self.llm.history = self.langchain_chat_history
            summary_res = self.llm("请对我们之前的对话内容进行总结。")
            print("请对我们之前的对话内容进行总结: ", summary_res)
            self.langchain_chat_history = []
            self.langchain_chat_history.append(("请对我们之前的对话内容进行总结。", summary_res))
            self.input_tokens = []
            self.input_tokens.append("请对我们之前的对话内容进行总结。")
            self.input_tokens.append(summary_res)
            return summary_res
        else:
            return ""
    
    def query_retrieval_llm(self, query, topk, prompt_type, prompt=None):
        new_query = self.get_new_question(query)
        user_prompt = self.create_user_query_prompt(new_query, topk, prompt_type, prompt)
        print("Post user query to EAS-LLM", user_prompt)
        self.llm.history = self.langchain_chat_history
        ans = self.llm(user_prompt)
        self.langchain_chat_history.append((new_query, ans))
        print("Get response from EAS-LLM.")
        self.input_tokens.append(new_query)
        self.input_tokens.append(ans)
        tokens_len = self.sp.encode(self.input_tokens, out_type=str)
        lens = sum(len(tl) for tl in tokens_len)
        summary_res = self.checkout_history_and_summary()
        return ans, lens, summary_res

    def query_only_llm(self, query):
        print("Post user query to EAS-LLM")
        start_time = time.time()
        self.llm.history = self.langchain_chat_history
        ans = self.llm(query)
        self.langchain_chat_history.append((query, ans))
        end_time = time.time()
        print("Get response from EAS-LLM. Cost time: {} s".format(end_time - start_time))
        self.input_tokens.append(query)
        self.input_tokens.append(ans)
        tokens_len = self.sp.encode(self.input_tokens, out_type=str)
        lens = sum(len(tl) for tl in tokens_len)
        summary_res = self.checkout_history_and_summary()
        return ans, lens, summary_res

    def query_only_vectorstore(self, query, topk):
        print("Post user query to Vectore Store")
        if topk == '' or topk is None:
            topk = 3
        start_time = time.time()
        print('query',query)
        docs = self.vector_db.similarity_search_db(query, topk=int(topk))

        page_contents, ref_names = [], []
        for idx, doc in enumerate(docs):
            content = doc.page_content if hasattr(doc, "page_content") else "[Doc Content Lost]"
            page_contents.append('='*20 + f' Doc [{idx+1}] ' + '='*20 + f'\n{content}\n')
            ref = doc.metadata['filename'] if hasattr(doc, "metadata") and "filename" in doc.metadata else "[Doc Name Lost]"
            ref_names.append(f'[{idx+1}] {ref}')

        ref_title = '='*20 + ' Reference Sources ' + '='*20
        context_docs = '\n'.join(page_contents) + f'{ref_title}\n' + '\n'.join(ref_names)
        end_time = time.time()
        print("Get response from Vectore Store. Cost time: {} s".format(end_time - start_time))
        tokens_len = self.sp.encode(context_docs, out_type=str)
        lens = sum(len(tl) for tl in tokens_len)
        return context_docs, lens
