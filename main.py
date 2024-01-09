import os
import shutil
import time

from app_modules.overwrites import postprocess
from app_modules.presets import *
from clc.config import LangChainCFG
from clc.langchain_application import LangChainApplication

config = LangChainCFG()
application = LangChainApplication(config)

application.source_service.init_source_vector()

def get_file_list():
    if not os.path.exists("docs"):
        return []
    return [f for f in os.listdir("docs")]


file_list = get_file_list()

def load_new_file(file, history):
    if file is None:
        msg_status = "没有收到上传文件"
    else:
        filename = os.path.basename(file.name)
        if os.path.exists("docs/" + filename):
            # file_list首位插入新上传的文件
            file_list.insert(0, filename)
            application.source_service.add_document("docs/" + filename)
            msg_status = f'{filename} 文件已成功加载'
        else:
            msg_status = f'{filename} 文件不存在'
    return '', history + [[None, msg_status]]

def upload_file(file):
    if not os.path.exists("docs"):
        os.mkdir("docs")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "docs/" + filename)
    # file_list首位插入新上传的文件
    #file_list.insert(0, filename)
    #application.source_service.add_document("docs/" + filename)
    return gr.Dropdown.update(choices=file_list, value=filename)


def set_knowledge(kg_name, history):
    try:
        application.source_service.load_vector_store(config.kg_vector_stores[kg_name])
        msg_status = f'{kg_name}知识库已成功加载'
    except Exception as e:
        print(e)
        msg_status = f'{kg_name}知识库未成功加载'
    return history + [[None, msg_status]]


def clear_session():
    return '', None

def show_user_msg(user_message, history):
    cur_history = history
    if history is None:
        cur_history = []
    cur_history = cur_history + [[user_message, None]]
    return user_message, cur_history

def predict(input,
            large_language_model,
            embedding_model,
            top_k,
            use_web,
            use_pattern,
            history=None):
    # print(large_language_model, embedding_model)
    print(input)
    if history == None:
        history = []

    if use_web == '使用':
        web_content = application.source_service.search_web(query=input)
    else:
        web_content = ''
    search_text = ''
    if use_pattern == '模型问答':
        result = application.get_llm_answer(query=input, web_content=web_content)
        history.append((input, result))
        if use_web == '使用':
            search_text += "----------【网络检索内容】-----------\n"
        search_text += web_content
        return '', history, search_text

    else:
        resp = application.get_knowledge_based_answer(
            query=input,
            history_len=1,
            temperature=0.1,
            top_p=0.9,
            top_k=top_k,
            web_content=web_content,
            chat_history=history
        )
        history.append((input, resp['result']))
        for idx, source in enumerate(resp['source_documents'][:4]):
            sep = f'----------【搜索结果{idx + 1}：】---------------\n'
            search_text += f'{sep}\n{source.page_content}\n\n'
        print(search_text)
        search_text += "----------【网络检索内容】-----------\n"
        search_text += web_content
        return '', history, search_text

# show stream response
def show_stream_response(history):
    if history is None:
        print("Exception history is None")
        return
    result = history[-1][1]
    message = history[-1][0]
    history.pop()
    history = history + [[message, '']]
    for char in result:
        history[-1][1] += char
        yield (history, history)
        time.sleep(0.03)

def change_model_type(large_language_model):
    application.change_model(large_language_model)
    return large_language_model

with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()
with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    gr.Markdown("""<h1><center>Chinese-LangChain-ChatGLM</center></h1>
        <center><font size=3>
        </center></font>
        """)
    state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            embedding_model = gr.Dropdown([
                "text2vec-base"
            ],
                label="Embedding model",
                value="text2vec-base")

            large_language_model = gr.Dropdown(
                [
                    config.chatglm_model_name,
                    config.qwen_model_name
                ],
                label="large language model",
                value=config.chatglm_model_name)

            top_k = gr.Slider(1,
                              20,
                              value=4,
                              step=1,
                              label="检索top-k文档",
                              interactive=True)

            use_web = gr.Radio(["使用", "不使用"], label="web search",
                               info="是否使用网络搜索，使用时确保网络通常",
                               value="不使用"
                               )
            use_pattern = gr.Radio(
                [
                    '模型问答',
                    '知识库问答',
                ],
                label="模式",
                value='模型问答',
                interactive=True)

            kg_name = gr.Radio(list(config.kg_vector_stores.keys()),
                               label="知识库",
                               value=None,
                               info="使用知识库问答，请加载知识库",
                               interactive=True)
            set_kg_btn = gr.Button("加载知识库")

            file = gr.File(label="将文件上传到知识库库，内容要尽量匹配",
                           visible=True,
                           file_types=['.txt', '.md', '.docx', '.pdf', 'doc']
                           )

            set_file_btn = gr.Button("加载新文件")

        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='Chinese-LangChain-ChatGLM').style(height=600)
            with gr.Row():
                message = gr.Textbox(label='请输入问题')
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")
            with gr.Row():
                gr.Markdown("""提醒：<br>
                                        [Chinese-LangChain-chatglm_cpp](https://github.com/icemoon/Chinese-LangChain-chatglm_cpp) <br>
                                        有任何使用问题[Github Issue区](https://github.com/icemoon/Chinese-LangChain-chatglm_cpp)进行反馈. <br>
                                        """)
        with gr.Column(scale=2):
            search = gr.Textbox(label='搜索结果')

        # ============= 触发动作=============
        # 下拉框修改模型
        large_language_model.change(change_model_type,
                                    inputs = [large_language_model],
                                    outputs= [large_language_model])

        file.upload(upload_file,
                    show_progress=True,
                    inputs=file,
                    outputs=None)

        set_file_btn.click(
            load_new_file,
            show_progress=True,
            inputs=[file, chatbot],
            outputs=[message, chatbot]
        )

        set_kg_btn.click(
            set_knowledge,
            show_progress=True,
            inputs=[kg_name, chatbot],
            outputs=chatbot
        )

        # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state],
                            queue=False)

        # 发送按钮 提交
        send.click(show_user_msg,
                    [message, state],
                    [message, chatbot],
                    queue=False).then(
        predict,
        [message, large_language_model, embedding_model,
        top_k, use_web, use_pattern, state],
        [message, state, search],
        queue=True).then(
            show_stream_response,
            state,
            [chatbot, state],
            queue=True
        )

        # 输入框 回车
        message.submit(show_user_msg,
                    [message, state],
                    [message, chatbot],
                    queue=False).then(
        predict,
        [message, large_language_model, embedding_model,
        top_k, use_web, use_pattern, state],
        [message, state, search],
        queue=True).then(
            show_stream_response,
            state,
            [chatbot, state],
            queue=True
        )

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    server_port=8888,
    share=False,
    show_error=True,
    debug=True,
    enable_queue=True,
    inbrowser=True,
)
