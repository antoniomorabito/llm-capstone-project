from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load vectorstore
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = db.as_retriever()

query = "Email policy"
docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(docs[:2]):
    print(f"Top {i+1}:\n{doc.page_content[:300]}\n")
