# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

from transformers import AutoModel, AutoTokenizer
import torch
from loguru import logger
import gc

class LocalLLM:
    model_path = ""
    max_length = 4096
    top_p = 0.8
    temperature = 0.7

    def __init__(self, model_name_or_path):
        self.model_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).eval().cuda()
        
        # 创建模型实例
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).eval()
        
        # 如果有多个GPU可用，使用DataParallel来包装模型
        if torch.cuda.device_count() > 1:
            logger.info(f"[RefGPT from {self.model_path}] Let's use multiple {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        else:
            logger.info(f"[RefGPT from {self.model_path}] Only use one {torch.cuda.device_count()} GPUs!")
        # 将模型发送到GPU（DataParallel将自动处理设备分配）
        
        self.model.cuda()
    
    def __call__(self, prompt: str):

        if hasattr(self.model, "chat"):
            response, _ = self.model.chat(self.tokenizer,
                                          prompt,
                                          history=[],
                                          max_length=self.max_length,
                                          temperature=self.temperature,
                                          top_p=self.top_p,
                                          do_sample=True)
        else:
            input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            input_len = input_ids['input_ids'].size(1)
            output = self.model.generate(**input_ids,
                                         temperature=self.temperature,
                                         top_p=self.top_p,
                                         max_length=self.max_length,
                                         do_sample=True)
            response = self.tokenizer.decode(output[0, input_len:])
        return response

    def del_model_cache(self):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            del self.model.module  # 删除模型的实际内容
        del self.model  # 删除模型的包装
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()