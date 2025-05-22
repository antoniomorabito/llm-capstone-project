from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

# Load document
loader = TextLoader("documents/new-Policies.txt")
docs = loader.load()

# Embedding
embeddings = OpenAIEmbeddings()

# Create VectorStore
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
db.persist()

# Query
query = "Smoking policy"
results = db.similarity_search(query, k=5)

for i, doc in enumerate(results):
    print(f"Result {i+1}:\n{doc.page_content[:200]}\n")
