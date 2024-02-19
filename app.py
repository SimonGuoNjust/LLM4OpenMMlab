from qaChain import Model_center
import gradio as gr

def init():
    model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b'
                                  , cache_dir='./', revision='v1.0.3')
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 下载模型
    os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir sentence-transformer')

def web_create():
    # 实例化核心功能对象
    model_center = Model_center()
    # 创建一个 Web 界面
    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):   
            with gr.Column(scale=15):
                # 展示的页面标题
                gr.Markdown("""<h1><center>InternLM</center></h1>
                    <center>书生浦语</center>
                    """)
    
        with gr.Row():
            with gr.Column(scale=4):
                # 创建一个聊天机器人对象
                chatbot = gr.Chatbot(height=450, show_copy_button=True)
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="Prompt/问题")
    
                with gr.Row():
                    # 创建提交按钮。
                    db_wo_his_btn = gr.Button("Chat")
                with gr.Row():
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                        components=[chatbot], value="Clear console")
                    
            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                                msg, chatbot], outputs=[msg, chatbot])
    
        gr.Markdown("""提醒：<br>
        1. 初始化数据库时间可能较长，请耐心等待。
        2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
        """)
    gr.close_all()
    # 直接启动
    demo.launch()

if __name__ == "__main__":
    init()
    web_create()