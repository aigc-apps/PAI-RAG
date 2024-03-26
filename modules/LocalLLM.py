# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

from transformers import AutoModel, AutoTokenizer
import torch
from loguru import logger
import gc
from utils.load_utils import *

class LocalLLM:
    model_path = ""
    max_length = 4096
    top_p = 0.8
    temperature = 0.7

    def __init__(self, model_name_or_path, device_map=None):
        self.model_path = model_name_or_path
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2 and device_map is None:
                logger.info(f"[RefGPT from {self.model_path}] using {num_gpus} GPUs!")
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
                model = (
                    AutoModel.from_pretrained(
                        self.model_path, trust_remote_code=True
                    )
                    .half()
                    .cuda()
                )

                if model and tokenizer:
                    self.model = model
                    self.tokenizer = tokenizer
                else:
                    raise Exception(
                        "faild to load " + self.model_path + " please try again."
                    )
                    return
            else:
                logger.info(f"[RefGPT from {self.model_path}][dispatch_model] using {num_gpus} GPUs!")
                from accelerate import dispatch_model
                logger.info(f'Using {num_gpus} GPUs!')
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
                model = (
                    AutoModel.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                    )
                    .half()
                    .cuda()
                )
                if model:
                    if device_map is None:
                        device_map = auto_configure_chatglm_device_map(num_gpus)
                    self.model = dispatch_model(model, device_map=device_map)
                    self.tokenizer = tokenizer
                else:
                    raise Exception(
                        "faild to load " + self.model_path + " please try again."
                    )
                    return
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            model = (
                AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                .half()
                .cuda()
            )
            if model:
                self.model = model
                self.tokenizer = tokenizer
            else:
                raise Exception(
                    "faild to load " + self.model_path + " please try again."
                )
                return

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