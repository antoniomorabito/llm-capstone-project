# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader

# # Load document
# loader = TextLoader("documents/new-Policies.txt")
# docs = loader.load()

# # Embedding
# embeddings = OpenAIEmbeddings()

# # Create VectorStore
# db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
# db.persist()

# # Query
# query = "Smoking policy"
# results = db.similarity_search(query, k=5)

# for i, doc in enumerate(results):
#     print(f"Result {i+1}:\n{doc.page_content[:200]}\n")


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Load text document
loader = TextLoader("documents/new-Policies.txt")
documents = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. Use sentence-transformers for embedding
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create vector database (Chroma)
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

# 5. Similarity search
query = "Smoking policy"
results = vectorstore.similarity_search(query, k=5)

# 6. Print results
for i, doc in enumerate(results, 1):
    print(f"Result {i}:\n{doc.page_content[:250]}\n{'-'*50}")
