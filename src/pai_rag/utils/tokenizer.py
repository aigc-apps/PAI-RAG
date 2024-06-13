import jieba
from nltk.corpus import stopwords
from typing import List

stopword_list = stopwords.words("chinese") + stopwords.words("english")


## PUT in utils file and add stopword in TRIE structure.
def jieba_tokenizer(text: str) -> List[str]:
    tokens = []
    for w in jieba.lcut(text):
        token = w.lower()
        if token not in stopword_list:
            tokens.append(token)

    return tokens
