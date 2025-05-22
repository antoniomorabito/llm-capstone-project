# 🔍 AI-Powered Document QA Bot using LangChain & Local Embeddings

This project is a Question Answering (QA) assistant that leverages LangChain, document loaders, vector databases, and local embedding models to answer questions based on PDF documents.

It was developed as part of the **LangChain + LLM Capstone Project** to simulate a Retrieval-Augmented Generation (RAG) assistant that works offline, without relying on paid APIs.

---

## 📌 Features

- ✅ Upload any PDF document
- ✅ Process and split content into vector chunks
- ✅ Use `sentence-transformers` for local embeddings
- ✅ Store vectors in FAISS or Chroma vector DB
- ✅ Retrieve relevant chunks using semantic search
- ✅ Simulate LLM answers using dummy logic or custom response
- ✅ Simple Gradio interface for real-time interaction

---

## 🧠 Stack Used

| Component           | Technology                  |
|---------------------|-----------------------------|
| Embeddings          | `sentence-transformers` (MiniLM) |
| Vector Store        | `FAISS` or `Chroma`         |
| LLM (simulated)     | Dummy LLM Response          |
| UI                  | `Gradio`                    |
| Framework           | `LangChain`                 |
| Language            | `Python 3.10+`              |

---

## 🚀 How to Run

1. **Clone the project**
   ```bash
   git clone https://github.com/your-repo/langchain-qa-bot
   cd langchain-qa-bot
2. Install dependencies
   pip install -r requirements.txt
   Run the QA bot interface

