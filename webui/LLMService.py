import json
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import AnalyticDB,Hologres,AlibabaCloudOpenSearch,AlibabaCloudOpenSearchSettings, ElasticsearchStore
import os
import nltk
import logging
import time
import requests
import sys
import argparse
import warnings
from chinese_text_splitter import ChineseTextSplitter
warnings.filterwarnings("ignore")

class LLMService:
    def __init__(self) -> None:
        self.cfg = None
        nltk_data_path = "/code/nltk_data"
        if os.path.exists(nltk_data_path):
            nltk.data.path = [nltk_data_path] + nltk.data.path

    def init_with_cfg(self,cfg):
        self.cfg = cfg
        self.vector_db, conn_time = self.connect_adb()
        return conn_time
     
    def post_to_chatglm2_eas(self, query_prompt):
        url = self.cfg['EASCfg']['url']
        token = self.cfg['EASCfg']['token']
        headers = {
            "Authorization": token,
            'Accept': "*/*",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        resp = requests.post(
            url=url,
            data=query_prompt.encode('utf8'),
            headers=headers,
            timeout=10000,
        )
        ans = json.loads(resp.text)
        return ans['response']
        
    def connect_adb(self):
        embedding_model = self.cfg['embedding']['embedding_model']
        model_dir = self.cfg['embedding']['model_dir']
        embed = HuggingFaceEmbeddings(model_name=os.path.join(model_dir, embedding_model), model_kwargs={'device': 'cpu'})
        emb_dim = self.cfg['embedding']['embedding_dimension']
        
        if 'ADBCfg' in self.cfg:
            start_time = time.time()
            pre_delete = 1 if self.cfg['ADBCfg']['PRE_DELETE']=='True' else 0
            connection_string_adb = AnalyticDB.connection_string_from_db_params(
                host=self.cfg['ADBCfg']['PG_HOST'],
                database=self.cfg['ADBCfg']['PG_DATABASE'],
                user=self.cfg['ADBCfg']['PG_USER'],
                password=self.cfg['ADBCfg']['PG_PASSWORD'],
                driver='psycopg2cffi',
                port=5432,
            )
            print('adb config pre_delete',pre_delete)
            vector_db = AnalyticDB(
                embedding_function=embed,
                embedding_dimension=emb_dim,
                connection_string=connection_string_adb,
                pre_delete_collection=int(pre_delete),
            )
            end_time = time.time()
            connect_time = end_time - start_time
            print("Connect AnalyticDB success. Cost time: {} s".format(connect_time))
        elif 'HOLOCfg' in self.cfg:
            start_time = time.time()
            connection_string_holo = Hologres.connection_string_from_db_params(
                host=self.cfg['HOLOCfg']['PG_HOST'],
                port=self.cfg['HOLOCfg']['PG_PORT'],
                database=self.cfg['HOLOCfg']['PG_DATABASE'],
                user=self.cfg['HOLOCfg']['PG_USER'],
                password=self.cfg['HOLOCfg']['PG_PASSWORD']
            )
            vector_db = Hologres(
                embedding_function=embed,
                ndims=emb_dim,
                connection_string=connection_string_holo,
            ) 
            end_time = time.time() 
            connect_time = end_time - start_time
            print("Connect Hologres success. Cost time: {} s".format(connect_time))
        elif 'ElasticSearchCfg' in self.cfg:
            start_time = time.time()
            vector_db = ElasticsearchStore(
                 es_url=self.cfg['ElasticSearchCfg']['ES_URL'],
                 index_name=self.cfg['ElasticSearchCfg']['ES_INDEX'],
                 es_user=self.cfg['ElasticSearchCfg']['ES_USER'],
                 es_password=self.cfg['ElasticSearchCfg']['ES_PASSWORD'],
                 embedding=embed
            )
            end_time = time.time() 
            connect_time = end_time - start_time
            print("Connect ElasticsearchStore success. Cost time: {} s".format(connect_time))
        return vector_db, connect_time
    
    def load_file(self,filepath,chunk_size,chunk_overlap):  
        if os.path.isdir(filepath):
            print('os.path.isdir')
            docs = DirectoryLoader(filepath).load()
            textsplitter = ChineseTextSplitter(
                pdf=False, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = textsplitter.split_documents(docs)
        elif filepath.lower().endswith(".md"):
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
        elif filepath.lower().endswith(".pdf"):
            print('os.path.pdf')
            loader = PyPDFLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = loader.load_and_split(textsplitter)
        else:
            print('os.path.else')
            loader = UnstructuredFileLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs
    
    def upload_custom_knowledge(self,docs_dir,chunk_size,chunk_overlap):
        # docs_dir = self.cfg['create_docs']['docs_dir']
        docs = self.load_file(docs_dir,chunk_size,chunk_overlap)
        # docs = DirectoryLoader(docs_dir, glob='**/*', show_progress=True).load()
        # text_splitter = CharacterTextSplitter(chunk_size=int(200), chunk_overlap=0)
        # docs = text_splitter.split_documents(docs)
        print('Uploading custom knowledge.')
        start_time = time.time()
        self.vector_db.add_documents(docs)
        end_time = time.time()
        print("Insert Success. Cost time: {} s".format(end_time - start_time))
        
    def create_user_query_prompt(self, query, topk, prompt):
        if topk == '' or topk is None:
            topk = 3
        docs = self.vector_db.similarity_search(query, k=int(topk))
        context_docs = ""
        for idx, doc in enumerate(docs):
            context_docs += "-----\n\n"+str(idx+1)+".\n"+doc.page_content
        context_docs += "\n\n-----\n\n"
        print('prompt',prompt)
        if prompt == '' or prompt is None:
            user_prompt_template = "基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 \"根据已知信息无法回答该问题\" 或 \"没有提供足够的相关信息\"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context}\n=====\n用户问题:\n{question}"
        else:
            user_prompt_template = prompt
        user_prompt_template = user_prompt_template.format(context=context_docs, question=query)
        source_docs = "\n--------------------\n Reference sources:" 
        for idx, doc in enumerate(docs):
            source_docs += "[" + str(idx+1)+"] " + doc.metadata['filename'] + ".    "
        return user_prompt_template, source_docs

    def user_query(self, query,topk,prompt):
        user_prompt_template, source_docs = self.create_user_query_prompt(query,topk,prompt)
        print("Post user query to EAS-LLM, user_prompt_template: ", user_prompt_template)
        start_time = time.time()
        ans = self.post_to_chatglm2_eas(user_prompt_template) + source_docs
        end_time = time.time()
        print("Get response from EAS-LLM. Cost time: {} s".format(end_time - start_time))
        return ans

    def query_only_llm(self,query):
        print("Post user query to EAS-LLM", query)
        start_time = time.time()
        ans = self.post_to_chatglm2_eas(query)
        end_time = time.time()
        print("Get response from EAS-LLM. Cost time: {} s".format(end_time - start_time))
        return ans

    def query_only_vectorestore(self,query,topk):
        print("Post user query to Vectore Store", query)
        start_time = time.time()
        if topk == '' or topk is None:
            topk = 3
        docs = self.vector_db.similarity_search(query, k=int(topk))
        context_docs = ""
        for idx, doc in enumerate(docs):
            context_docs += str(idx+1)+". "+doc.page_content + "\n"
        context_docs += "\n-----\n Reference sources:" 
        for idx, doc in enumerate(docs):
            context_docs += "[" + str(idx+1)+"] " + doc.metadata['filename'] + ".    "
        end_time = time.time()
        print("Get response from Vectore Store. Cost time: {} s".format(end_time - start_time))
        return context_docs