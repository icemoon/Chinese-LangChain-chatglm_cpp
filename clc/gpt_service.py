#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: generate.py
@time: 2023/04/17
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import os
from typing import Dict, Union, Optional
from typing import List


from accelerate import load_checkpoint_and_dispatch
from clc.config import LangChainCFG
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.chains import LLMChain
from langchain.llms import ChatGLM
from langchain.llms import LlamaCpp

from transformers import AutoModel, AutoTokenizer

import subprocess
import signal
import sys

class ChatGLMService(LLM):
    config: object = None
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    chatglm_engine: object = None
    chatglm_app: str = None
    llama_engine: object = None
    llama_model_name: str = None
    # LLama, ChatGLM
    active_engine_name: str = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chatglm_app = "chatglm_cpp.langchain_api:app --host 127.0.0.1 --port 8000"
        self.active_engine_name = config.chatglm_engine_name

    @property
    def _llm_type(self) -> str:
        return self.active_engine_name

    def call_chatglm(self,
                    prompt: str,
                    stop: Optional[List[str]] = None) -> str:
        response = self.chatglm_engine._call(prompt, stop)
        return response

    def call_llama(self,
                    prompt: str,
                    stop: Optional[List[str]] = None) -> str:
        prompt_prefix = "Question:"
        prompt_suffix = " Answer: "
        prompt_stop = [prompt_prefix]
        if stop is not None:
            prompt_stop = stop
        prompt = prompt_prefix + prompt + prompt_suffix
        response = self.llama_engine._call(prompt, prompt_stop)
        return response

    def change_model(self, model_name):
        cur_active_engine_name = self.config.model_engines[model_name]
        if cur_active_engine_name == self.config.llama_engine_name and self.llama_model_name != model_name:
            del self.llama_engine
            if self.load_llama_mode(self.config.model_files[model_name]) == False:
                sys.exit()
            self.llama_model_name = model_name
        self.active_engine_name = cur_active_engine_name
        print("Change model, active_engine is %s, llama model is %s"
            % (self.active_engine_name, self.llama_model_name))

    def _call(self,
            prompt: str) -> str:
        ### ypy code
        if self.active_engine_name == self.config.chatglm_engine_name:
            return self.call_chatglm(prompt)
        else:
            return self.call_llama(prompt)

    def start_chatglm(self, model_name):
        start_cmd = "MODEL=" + model_name + " uvicorn " + self.chatglm_app
        p = subprocess.Popen(start_cmd,
                        shell=True,
                        stdout=sys.stdout,
                        stderr=sys.stdout)
        print("Start chatglm pid: ", p.pid)

    def check_chatglm(self):
        check_cmd = "ps aux | grep python | grep -v grep | grep \'" + self.chatglm_app + "\'"
        p = subprocess.Popen(check_cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True)
        p.wait()
        check_out = p.stdout.read().strip()
        if check_out == '':
            print("Check: Chatglm not running")
            return False
        else:
            print("Check: Chatglm running as below\n" + check_out)
            return True

    # load chatglm model ggml file
    def load_chatglm_model(self, model_name_or_path):
        if (self.check_chatglm() == False):
            self.start_chatglm(model_name_or_path)
            if (self.check_chatglm() == False):
                print("Load model fail: " + model_name_or_path)
                return False
        self.chatglm_engine = ChatGLM(endpoint_url="http://127.0.0.1:8000")
        return True

    # load chatglm model gguf file
    def load_llama_mode(self, model_name_or_path):
        self.llama_engine = LlamaCpp(
                model_path = model_name_or_path,
                temperature = self.temperature,
                max_tokens = self.max_token,
                n_ctx = self.max_token,
                top_p = self.top_p)
        return True

    def load_model(self):
        ### ypy code
        self.llama_model_name = self.config.qwen_model_name
        if self.load_llama_mode(self.config.model_files[self.llama_model_name]) == False:
            sys.exit()
        if self.load_chatglm_model(self.config.model_files[self.config.chatglm_model_name]) == False:
            sys.exit()
        print("Load model finish, active_engine is %s, llama model is %s"
            % (self.active_engine_name, self.llama_model_name))

# if __name__ == '__main__':
#     config=LangChainCFG()
#     chatLLM = ChatGLMService()
#     chatLLM.load_model(model_name_or_path=config.llm_model_name)
