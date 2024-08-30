import jieba
from nltk.corpus import stopwords
from typing import List
from pai_rag.utils.trie import TrieTree
import string

CHINESE_PUNKTUATION = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。"

stopword_list = stopwords.words("chinese") + stopwords.words("english")
stopword_list += [" "] + list(string.punctuation) + list(CHINESE_PUNKTUATION)
stop_trie = TrieTree(stopword_list)


## PUT in utils file and add stopword in TRIE structure.
def jieba_tokenizer(text: str) -> List[str]:
    tokens = []
    for w in jieba.cut(text):
        token = w.lower()
        if not stop_trie.match(token):
            tokens.append(token)

    return tokens
