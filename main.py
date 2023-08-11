import json
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import AnalyticDB
import os
import logging
import time
import requests
import sys
import argparse

class LLMService:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.vector_db = self.connect_adb()
    
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
        return resp.text
        
    
    def connect_adb(self):
        connection_string = AnalyticDB.connection_string_from_db_params(
            host=self.cfg['ADBCfg']['PG_HOST'],
            database='postgres',
            user=self.cfg['ADBCfg']['PG_USER'],
            password=self.cfg['ADBCfg']['PG_PASSWORD'],
            driver='psycopg2cffi',
            port=5432,
        )
        
        embedding_model = self.cfg['embedding']['embedding_model']
        model_dir = self.cfg['embedding']['model_dir']
        embed = HuggingFaceEmbeddings(model_name=os.path.join(model_dir, embedding_model), model_kwargs={'device': 'cpu'})
        
        vector_db = AnalyticDB(
            embedding_function=embed,
            embedding_dimension=cfg['embedding']['embedding_dimension'],
            connection_string=connection_string,
            # pre_delete_collection=self.is_delete,
        )
        return vector_db
    
    def upload_custom_knowledge(self):
        docs_dir = self.cfg['create_docs']['docs_dir']
        docs = DirectoryLoader(docs_dir, glob=self.cfg['create_docs']['glob'], show_progress=True).load()
        text_splitter = CharacterTextSplitter(chunk_size=int(self.cfg['create_docs']['chunk_size']), chunk_overlap=self.cfg['create_docs']['chunk_overlap'])
        docs = text_splitter.split_documents(docs)
        start_time = time.time()
        self.vector_db.add_documents(docs)
        end_time = time.time()
        print("Insert into AnalyticDB Success. Cost time: {} s".format(end_time - start_time))
        
    def create_user_query_prompt(self, query):
        docs = self.vector_db.similarity_search(query, k=int(cfg['query_topk']))
        context_docs = ""
        for idx, doc in enumerate(docs):
            context_docs += "-----\n\n"+str(idx+1)+".\n"+doc.page_content
        context_docs += "\n\n-----\n\n"
        user_prompt_template = self.cfg['prompt_template']
        user_prompt_template = user_prompt_template.format(context=context_docs, question=query)
        return user_prompt_template

    def user_query(self, query):
        user_prompt_template = self.create_user_query_prompt(query)
        print("Post user query to EAS-ChatGLM2")
        start_time = time.time()
        ans = self.post_to_chatglm2_eas(user_prompt_template)
        end_time = time.time()
        print("Get response from EAS-ChatGLM2. Cost time: {} s".format(end_time - start_time))
        return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line argument parser')
    parser.add_argument('--config', type=str, help='json配置文件输入', default='config.json')
    parser.add_argument('--upload', help='上传知识库', default=False)
    parser.add_argument('--query', help='用户请求查询')
    args = parser.parse_args()
    if args.config:
        if not args.upload and not args.query:
                print('Not any operation is set.')
        else:
            if os.path.exists(args.config):   
                with open(args.config) as f:
                    cfg = json.load(f)
                    solver = LLMService(cfg)
                    if args.upload:
                        print('Uploading custom files to ADB.')
                        solver.upload_custom_knowledge()
                        
                    if args.query:
                        user_prompt_template = solver.create_user_query_prompt(args.query)
                        print('args.query',args.query)
                        answer = solver.user_query(args.query)
                        print("The answer is: ", answer)
            else:
                print(f"{args.config} does not exist.")
    else:
        print("The config json file must be set.")