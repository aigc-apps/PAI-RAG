import json
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from .PromptTemplate import PromptTemplate
from .EASAgent import EASAgent
from .VectorDB import VectorDB
import time

class LLMService:
    def __init__(self, args):
        self.prompt_template = PromptTemplate(args)

        assert args.upload or args.query, "error: dose not set any action, please set '--upload' or '--query <user_query>'."
        assert os.path.exists(args.config), f"error: config path {args.config} does not exist."

        with open(args.config) as f:
            self.cfg = json.load(f)
            self.eas_agent = EASAgent(self.cfg, args)
            self.vector_db = VectorDB(args, self.cfg)
            if args.upload:
                self.upload_custom_knowledge()
            if args.query:
                answer = self.get_answer(args.query)
                print("The answer is: ", answer)

    def upload_custom_knowledge(self):
        docs_dir = self.cfg['create_docs']['docs_dir']
        docs = DirectoryLoader(docs_dir, glob=self.cfg['create_docs']['glob'], show_progress=True).load()
        text_splitter = CharacterTextSplitter(chunk_size=int(self.cfg['create_docs']['chunk_size']), chunk_overlap=self.cfg['create_docs']['chunk_overlap'])
        docs = text_splitter.split_documents(docs)

        print('Uploading custom knowledge.')
        start_time = time.time()
        self.vector_db.add_documents(docs)
        end_time = time.time()
        print("Insert Success. Cost time: {} s".format(end_time - start_time))

    def create_user_query_prompt(self, query):
        docs = self.vector_db.similarity_search(query)
        user_prompt = self.prompt_template.get_prompt(docs, query)

        return user_prompt

    def get_answer(self, query):
        user_prompt = self.create_user_query_prompt(query)
        print("Post user query to EAS-LLM")
        start_time = time.time()
        ans = self.eas_agent.post_to_eas(user_prompt)
        end_time = time.time()
        print("Get response from EAS-LLM. Cost time: {} s".format(end_time - start_time))

        return ans
