import gradio as gr
from document_analysis import DocumentAnalysis
from config import llm_models, knowledge_path
from loguru import logger
import signal
import sys
import os

# Initialize document analysis object
document_analysis_obj = DocumentAnalysis()


def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM"""
    logger.info("Shutting down gracefully...")
    try:
        # Cleanup document analysis resources
        document_analysis_obj.cleanup()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
    finally:
        sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def submit(query, chat_history):
    logger.info(f'query: {query}')
    logger.info(f'chat_history: {chat_history}')

    # 如果查询为空字符串，返回空字符串和当前的聊天记录
    if query == '':
        return '', chat_history

    # 确保chat_history是列表格式
    if not isinstance(chat_history, list):
        chat_history = []
    
    # 添加新的问题和空的回答
    chat_history.append([query, None])

    # 确保chat_history不超过6对对话
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]

    return '', chat_history


async def change_collection():
    """ 在更改知识库时清除模型历史记录"""
    document_analysis_obj.clear_history()


async def load_history():
    """ 加载模型的历史消息 """
    # 将历史消息格式化为对话形式，每两条消息为一对
    history = document_analysis_obj.get_history_message()
    return history


async def llm_reply(collection_names, chat_history, model=None, max_length=512, temperature=0.1):
    """ 生成模型回复 """
    if not chat_history or not isinstance(chat_history, list):
        logger.error("Invalid chat history format")
        return []

    logger.info(f"Selected collections: {collection_names}")
    
    try:
        # 获取最后一个问题
        if len(chat_history) == 0 or not isinstance(chat_history[-1], list):
            logger.error("Invalid chat history format")
            return chat_history

        question = chat_history[-1][0]
        if not question or not isinstance(question, str):
            logger.error("Invalid question format")
            return chat_history
        
        # 使用流式生成回答
        answer = ""
        # 如果没有选择知识库，直接使用大语言模型回答
        if not collection_names or (isinstance(collection_names, list) and len(collection_names) == 0):
            logger.info("No knowledge base selected, using LLM directly")
            async for chunk in document_analysis_obj.stream(question, [], model, max_length, temperature):
                if chunk and 'answer' in chunk:
                    answer += chunk['answer']
        else:
            # 使用知识库进行回答
            async for chunk in document_analysis_obj.stream(question, collection_names, model, max_length, temperature):
                if chunk and 'answer' in chunk:
                    answer += chunk['answer']
        
        # 更新最后一个对话的回答
        chat_history[-1][1] = answer
        return chat_history

    except Exception as e:
        logger.error(f"Error in llm_reply: {str(e)}")
        if len(chat_history) > 0:
            chat_history[-1][1] = "抱歉，处理请求时出现错误"
        return chat_history


async def init_knowledge():
    """初始化知识库"""
    try:
        # 清理历史记录
        document_analysis_obj.clear_history()
        return []
    except Exception as e:
        logger.error(f"Error initializing: {str(e)}")
        return []


with gr.Blocks(fill_height=True) as demo:
    """ 创建一个Gradio Blocks应用，设置fill_height为True """
    # 在应用中添加一个HTML元素，显示标题
    gr.HTML("""<h1 align="center">通用文档分析系统</h1>""")

    # 创建一个新的行布局
    with gr.Row():
        # 创建一个占比为 4 的列布局
        with gr.Column(scale=4):
            # 创建一个下拉菜单，用于选择LLM模型，默认为 "gpt-4o"
            model = gr.Dropdown(
                choices=llm_models,
                value=llm_models[1],
                label="Openai LLM Model",
                interactive=True,
                scale=1
            )

            # 创建一个聊天机器人界面
            chatbot = gr.Chatbot(
                show_label=False, 
                scale=3, 
                show_copy_button=True,
                height=600,
                bubble_full_width=False
            )

        # 创建一个占比为 1 的列布局，显示进度
        with gr.Column(scale=1, show_progress=True) as column_config:
            # 创建一个滑块，用于设置生成回复的最大长度
            max_length = gr.Slider(1, 4095, value=512, step=1.0, label="Maximum length", interactive=True)
            # 创建一个滑块，用于设置生成回复的温度
            temperature = gr.Slider(0, 2, value=0.1, step=0.01, label="Temperature", interactive=True)
            # 创建一个按钮，用于清除聊天记录
            clear = gr.Button("清除")
            # 创建一个多选下拉菜单，用于选择知识库
            collection = gr.Dropdown(
                label="知识库",
                choices=[],  # 初始为空
                value=[],  # 初始没有选择
                interactive=True,
                multiselect=True,  # 允许多选
                allow_custom_value=False  # 不允许自定义值
            )

            # 创建一个文件上传控件，支持多种文件类型
            file = gr.File(
                label="上传文件",
                file_types=['.doc', '.docx', '.txt', '.pdf', '.md'],
                file_count="single"
            )

    # 创建一个文本框，用于用户输入
    user_input = gr.Textbox(placeholder="输入...", show_label=False)

    # 创建一个按钮，用于提交用户输入
    user_submit = gr.Button("提交")

    # 绑定 clear 按钮的点击事件，清除模型历史记录，并更新聊天机器人界面
    clear.click(fn=document_analysis_obj.clear_history, inputs=None, outputs=[chatbot])

    # 回车提交
    user_input.submit(fn=submit,
                     inputs=[user_input, chatbot],
                     outputs=[user_input, chatbot]
                     ).then(
        fn=llm_reply,
        inputs=[collection, chatbot, model, max_length, temperature],
        outputs=[chatbot]
    )

    # 提交按钮提交
    user_submit.click(fn=submit,
                     inputs=[user_input, chatbot],
                     outputs=[user_input, chatbot]
                     ).then(
        fn=llm_reply,
        inputs=[collection, chatbot, model, max_length, temperature],
        outputs=[chatbot]
    )

    # 绑定文件上传控件的上传事件，调用upload_knowledge函数，并更新文件控件和知识库下拉菜单
    file.upload(fn=document_analysis_obj.upload_knowledge, inputs=[file], outputs=[file, collection])

    # 绑定知识库下拉菜单的更改事件，调用clear_history函数，并更新聊天机器人界面
    collection.change(fn=document_analysis_obj.clear_history, inputs=None, outputs=[chatbot])

    # 绑定应用加载事件，加载已有知识库和清除历史记录
    def init_ui():
        """初始化UI"""
        # 同步获取文件列表
        files = []
        if os.path.exists(knowledge_path):
            files = [f for f in os.listdir(knowledge_path)
                    if os.path.isfile(os.path.join(knowledge_path, f))]
            files = [str(f).strip() for f in files]
            files = sorted([f for f in files if f])
        return gr.update(choices=files, value=[])  # 返回空列表作为初始值

    demo.load(
        fn=init_ui,
        outputs=collection
    ).then(
        fn=document_analysis_obj.clear_history,
        inputs=None,
        outputs=[chatbot]
    )

if __name__ == "__main__":
    # 启动 Gradio 应用
    demo.queue().launch()
