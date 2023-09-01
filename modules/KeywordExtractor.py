# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

import jieba
from keybert import KeyBERT

class KeywordExtractor:
    def __init__(self, args):
        pass

    def keywords_textrank(self, doc):
        keywords = jieba.analyse.extract_tags(query, topK=5, withWeight=False, allowPOS=('n', 'ns', 'v', 'a', 'eng'))
        query_with_keywords = ' '.join(keywords) + ' ' + query
        print(query_with_keywords)
        query_with_keywords_list = []
        query_with_keywords_list.append(query_with_keywords)

    def keywords_keybert(self, doc):
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(doc)
        key_words = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)