# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""an example of a QA robot based on PAI documents

langchain modules:
    db: Hologres
    LLM: Qwen-7B
    text splitter: ChineseTextSplitter_V2
    embedding model: SGPT-5B
    key_words model: textrank
    prompt engineering: Retrieval-Augmented Generation

"""
from modules.LLMService import LLMService
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line argument parser')
    parser.add_argument('--config',
                        type=str,
                        help='json配置文件输入',
                        required=True,
                        default='config.json')
    parser.add_argument('--prompt_engineering',
                        type=str,
                        help='prompt模板类型',
                        choices=['Retrieval-Augmented Generation'],
                        default='Retrieval-Augmented Generation')
    parser.add_argument('--embed_model',
                        type=str,
                        required=True,
                        help='embedding模型名称',
                        default='SGPT-125M-weightedmean-nli-bitfit')
    parser.add_argument('--embed_dim',
                        type=int,
                        help='embedding向量维度',
                        default=768)
    parser.add_argument('--upload',
                        action='store_true',
                        help='上传知识库',
                        default=False)
    parser.add_argument('--query',
                        help='用户请求查询')
    args = parser.parse_args()

    solver = LLMService(args)
