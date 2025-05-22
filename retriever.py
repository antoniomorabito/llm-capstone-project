# # from langchain.vectorstores import Chroma
# # from langchain.embeddings import OpenAIEmbeddings

# # # Load vectorstore
# # embeddings = OpenAIEmbeddings()
# # db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# # retriever = db.as_retriever()

# # query = "Email policy"
# # docs = retriever.get_relevant_documents(query)

# # for i, doc in enumerate(docs[:2]):
# #     print(f"Top {i+1}:\n{doc.page_content[:300]}\n")


# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings

# # 1. Load and prepare document
# loader = TextLoader("documents/new-Policies.txt")
# documents = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = splitter.split_documents(documents)

# # 2. Embedding model
# embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # 3. Load Chroma Vector Store
# vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

# # 4. Convert to retriever
# retriever = vectorstore.as_retriever()

# # 5. Query
# query = "Email policy"
# results = retriever.get_relevant_documents(query)

# # 6. Print top 2 results
# for i, doc in enumerate(results[:2], 1):
#     print(f"Top {i} Result:\n{doc.page_content[:250]}\n{'-'*50}")



from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Load and prepare document
loader = TextLoader("documents/new-Policies.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 2. Embedding model
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Load Chroma Vector Store
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

# 4. Convert to retriever
retriever = vectorstore.as_retriever()

# 5. Query
query = "Email policy"
results = retriever.get_relevant_documents(query)

# 6. Print top 2 results
for i, doc in enumerate(results[:2], 1):
    print(f"Top {i} Result:\n{doc.page_content[:250]}\n{'-'*50}")
