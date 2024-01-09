#!/usr/bin/env python
# -*- coding:utf-8 _*-

# 本地模型文件, 修改成自己的配置！！！
chatglm_model_path = 'path/chatglm3-ggml-q4_0.bin'
qwen_model_path = 'path/Qwen-7B-Chat.Q4_K_M.gguf'

class LangChainCFG:
    embedding_model_name = 'path/text2vec-large-chinese' # 检索模型文件 or huggingface远程仓库
    chatglm_model_name = 'ChatGLM-6B-int4'
    qwen_model_name = 'Qwen-7B-int4'
    chatglm_engine_name = 'ChatGLM'
    llama_engine_name = 'LLama'
    model_engines = {
        chatglm_model_name: chatglm_engine_name,
        qwen_model_name: llama_engine_name,
    }
    model_files = {
        chatglm_model_name: chatglm_model_path,
        qwen_model_name: qwen_model_path,
    }
    vector_store_path = './cache'
    docs_path = './docs'
    kg_vector_stores = {
        '中文维基百科': './cache/zh_wikipedia',
        '大规模金融研报': './cache/financial_research_reports',
        '初始化': './cache',
    }  # 可以替换成自己的知识库，如果没有需要设置为None
    # kg_vector_stores=None
    patterns = ['模型问答', '知识库问答']  #
    n_gpus=1