from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Load local model
model_name = "llama3.2"  # ringan dan cepat
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name
)

# Kalimat untuk di-embed
query = "How are you?"
embedding = embedding_function.embed_query(query)

# Tampilkan 5 angka pertama
print("First 5 values of embedding:")
print(embedding[:5])
