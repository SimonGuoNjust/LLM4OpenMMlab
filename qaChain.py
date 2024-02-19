__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from LLM import InternLM_LLM
import os

def load_chain():
    
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformer")
    
    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    
    llm = InternLM_LLM(model_path = "Shanghai_AI_Laboratory/internlm-chat-7b")
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
    
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    return qa_chain, llm
    
class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain, _ = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history

if __name__ == "__main__":
    qa_chain, llm = load_chain()
    # 检索问答链回答效果
    question = "什么是InternLM"
    result = qa_chain({"query": question})
    print("检索问答链回答 question 的结果：")
    print(result["result"])
    
    # 仅 LLM 回答效果
    result_2 = llm(question)
    print("大模型回答 question 的结果：")
    print(result_2)

