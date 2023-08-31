import json
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import AnalyticDB,Hologres,AlibabaCloudOpenSearch,AlibabaCloudOpenSearchSettings
from langchain.vectorstores import ElasticsearchStore
import os
import logging
import time
import requests
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

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
        embedding_model = self.cfg['embedding']['embedding_model']
        model_dir = self.cfg['embedding']['model_dir']
        self.embed = HuggingFaceEmbeddings(model_name=os.path.join(model_dir, embedding_model), model_kwargs={'device': 'cpu'})
        emb_dim = cfg['embedding']['embedding_dimension']

        if 'ADBCfg' in self.cfg:
            start_time = time.time()
            connection_string_adb = AnalyticDB.connection_string_from_db_params(
                host=self.cfg['ADBCfg']['PG_HOST'],
                database='postgres',
                user=self.cfg['ADBCfg']['PG_USER'],
                password=self.cfg['ADBCfg']['PG_PASSWORD'],
                driver='psycopg2cffi',
                port=5432,
            )
            vector_db = AnalyticDB(
                embedding_function=self.embed,
                embedding_dimension=emb_dim,
                connection_string=connection_string_adb,
                # pre_delete_collection=True,
            )
            end_time = time.time()
            print("Connect AnalyticDB success. Cost time: {} s".format(end_time - start_time))
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
                embedding_function=self.embed,
                ndims=emb_dim,
                connection_string=connection_string_holo,
            )
            end_time = time.time()
            print("Connect Hologres success. Cost time: {} s".format(end_time - start_time))
        elif 'ElasticSearchCfg' in self.cfg:
            start_time = time.time()
            vector_db = ElasticsearchStore(
                 es_url=self.cfg['ElasticSearchCfg']['ES_URL'],
                 index_name=self.cfg['ElasticSearchCfg']['ES_INDEX'],
                 es_user=self.cfg['ElasticSearchCfg']['ES_USER'],
                 es_password=self.cfg['ElasticSearchCfg']['ES_PASSWORD'],
                 embedding=self.embed
            )
            end_time = time.time()
            print("Connect ElasticsearchStore success. Cost time: {} s".format(end_time - start_time))
        elif 'OpenSearchCfg' in self.cfg:
            start_time = time.time()
            print("Start Connect AlibabaCloudOpenSearch ")
            settings = AlibabaCloudOpenSearchSettings(
                endpoint=self.cfg['OpenSearchCfg']['endpoint'],
                instance_id=self.cfg['OpenSearchCfg']['instance_id'],
                datasource_name=self.cfg['OpenSearchCfg']['datasource_name'],
                username=self.cfg['OpenSearchCfg']['username'],
                password=self.cfg['OpenSearchCfg']['password'],
                embedding_index_name=self.cfg['OpenSearchCfg']['embedding_index_name'],
                field_name_mapping={
                    "id": self.cfg['OpenSearchCfg']['field_name_mapping']['id'],
                    "document": self.cfg['OpenSearchCfg']['field_name_mapping']['document'],
                    "embedding": self.cfg['OpenSearchCfg']['field_name_mapping']['embedding'],
                    "source": self.cfg['OpenSearchCfg']['field_name_mapping']['source'],
                },
            )
            vector_db = AlibabaCloudOpenSearch(
                embedding=self.embed, config=settings
            )
            end_time = time.time()
            print("Connect AlibabaCloudOpenSearch success. Cost time: {} s".format(end_time - start_time))
        else:
            print("Not config any database, use FAISS-cpu default.")
            vector_db = None
        return vector_db

    def upload_custom_knowledge(self):
        docs_dir = self.cfg['create_docs']['docs_dir']
        docs = DirectoryLoader(docs_dir, glob=self.cfg['create_docs']['glob'], show_progress=True).load()
        text_splitter = CharacterTextSplitter(chunk_size=int(self.cfg['create_docs']['chunk_size']), chunk_overlap=self.cfg['create_docs']['chunk_overlap'])
        docs = text_splitter.split_documents(docs)
        print('Uploading custom knowledge.')
        start_time = time.time()
        if all(item not in self.cfg for item in ['ADBCfg','HOLOCfg','ElasticSearchCfg','OpenSearchCfg']):
            self.vector_db = FAISS.from_documents(docs,self.embed)
            self.vector_db.save_local("faiss_index")
        else:
            self.vector_db.add_documents(docs)
        end_time = time.time()
        print("Insert Success. Cost time: {} s".format(end_time - start_time))

    def create_user_query_prompt(self, query):
        if all(item not in self.cfg for item in ['ADBCfg','HOLOCfg','ElasticSearchCfg','OpenSearchCfg']):
            self.vector_db = FAISS.load_local("faiss_index", self.embed)
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
        print("Post user query to EAS-LLM")
        start_time = time.time()
        ans = self.post_to_chatglm2_eas(user_prompt_template)
        end_time = time.time()
        print("Get response from EAS-LLM. Cost time: {} s".format(end_time - start_time))
        return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line argument parser')
    parser.add_argument('--config', type=str, help='json配置文件输入', default='config.json')
    parser.add_argument('--upload', action='store_true', help='上传知识库', default=False)
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
                        solver.upload_custom_knowledge()
                    if args.query:
                        answer = solver.user_query(args.query)
                        print("The answer is: ", answer)
            else:
                print(f"{args.config} does not exist.")
    else:
        print("The config json file must be set.")
