# === embedding.py ===

from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file

# === Use Watsonx if available ===
# from langchain.embeddings import WatsonxEmbeddings
# embeddings = WatsonxEmbeddings(
#     model_id="watsonx/embedding-model-id",
#     watsonx_api_key=os.getenv("WATSONX_API_KEY")
# )

# === Fallback to OpenAI ===
from langchain.embeddings import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Query to embed
query = "How are you?"
embedding = embedding_function.embed_query(query)

print("First 5 values of embedding:")
print(embedding[:5])
