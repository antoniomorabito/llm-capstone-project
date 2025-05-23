# # === qa_bot.py ===

# from dotenv import load_dotenv
# import os
# load_dotenv()

# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# import gradio as gr

# # === Watsonx LLM Setup (simulasi struktur) ===
# # from langchain.llms import WatsonxLLM
# # llm = WatsonxLLM(
# #     model_id="mistralai/mixtral-8x7b-instruct-v01",
# #     watsonx_api_key=os.getenv("WATSONX_API_KEY"),
# #     project_id=os.getenv("WATSONX_PROJECT_ID")
# # )

# # === Fallback LLM: OpenAI GPT-3.5 ===
# from langchain.chat_models import ChatOpenAI
# llm = ChatOpenAI(
#     temperature=0,
#     model_name="gpt-3.5-turbo",
#     openai_api_key=os.getenv("OPENAI_API_KEY")
# )

# # === Embedding ===
# from langchain.embeddings import OpenAIEmbeddings
# embedding_function = OpenAIEmbeddings(
#     openai_api_key=os.getenv("OPENAI_API_KEY")
# )

# def build_qa_bot(file, query):
#     # 1. Load PDF
#     loader = PyPDFLoader(file.name)
#     docs = loader.load()

#     # 2. Split
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = splitter.split_documents(docs)

#     # 3. Embed + Store in Vector DB
#     db = FAISS.from_documents(chunks, embedding_function)

#     # 4. Retrieval + QA
#     retriever = db.as_retriever()
#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#     # 5. Run
#     return qa.run(query)

# # === Gradio Frontend ===
# iface = gr.Interface(
#     fn=build_qa_bot,
#     inputs=[gr.File(type="file"), gr.Textbox(label="Ask a question")],
#     outputs="text",
#     title="QA Bot Powered by LLM"
# )

# iface.launch()


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import gradio as gr
import random

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Dummy "LLM" that gives fake answer
def dummy_llm_answer(query, context):
    responses = [
        "This paper discusses applications of large language models in document processing.",
        "The content focuses on retrieval-augmented generation and how to apply QA over PDFs.",
        "It highlights challenges in analyzing long-form scientific documents."
    ]
    return random.choice(responses)

def qa_from_pdf(file, query):
    loader = PyPDFLoader(file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embedding_function)
    retriever = db.as_retriever()
    top_docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in top_docs])
    return dummy_llm_answer(query, context)

iface = gr.Interface(
    fn=qa_from_pdf,
    inputs=[gr.File(type="filepath"), gr.Textbox(label="Ask a question")],
    outputs="text",
    title="Offline QA Bot"
)

iface.launch()
