# === QA BOT - Watsonx-Compatible Version ===

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import gradio as gr

# === Watsonx.ai LLM Setup ===
# from langchain.llms import WatsonxLLM   # ← Uncomment this line if Watsonx SDK is installed
# llm = WatsonxLLM(
#     model_id="mistralai/mixtral-8x7b-instruct-v01",
#     watsonx_api_key="your_watsonx_api_key",
#     watsonx_project_id="your_project_id"
# )

# === Dummy fallback using OpenAI for testing ===
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
os.environ["OPENAI_API_KEY"] = "your_openai_key"
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
embedding_function = OpenAIEmbeddings()

def build_qa_bot(file, query):
    # 1. Load PDF
    loader = PyPDFLoader(file.name)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3. Generate embeddings (fallback to OpenAI here)
    db = FAISS.from_documents(chunks, embedding_function)

    # 4. Create retriever + QA chain
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 5. Run query
    result = qa.run(query)
    return result

# === Gradio Frontend ===
iface = gr.Interface(
    fn=build_qa_bot,
    inputs=[gr.File(type="file"), gr.Textbox(label="Ask a question")],
    outputs="text",
    title="Watsonx QA Bot (Simulated)"
)

iface.launch()
