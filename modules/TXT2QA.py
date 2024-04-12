from utils.generator import TxtGenerator
from loguru import logger

class TXT2QA:
    def __init__(self, config):
        self.config = config['TXTCfg']
        self.generator = TxtGenerator(self.config)
    
    def run(self, docs):
        result = []
        for doc in docs:
            logger.info(f"generating qa pairs for doc:\n{doc}\n\n")
            text_content = doc.page_content
            qa_dict = self.generator.generate_qa(text_content)
            
            result.append(qa_dict)
        
        return result

    def del_model_cache(self):
        if self.config['LLM'] == 'Local':
            logger.info("Removing local llm cache from gpu memory.")
            self.generator.llm.del_model_cache()
            logger.info("Clear finished.")

if __name__ == "__main__":
    x = TXT2QA()
    x.run()