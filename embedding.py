# === Embedding with Watsonx-Compatible Setup ===

# from langchain.embeddings import WatsonxEmbeddings  # Uncomment if SDK installed
# embeddings = WatsonxEmbeddings(
#     model_id="watsonx/embedding-model-id",
#     watsonx_api_key="your_watsonx_api_key"
# )

# === Dummy fallback with OpenAI ===
from langchain.embeddings import OpenAIEmbeddings
import os
os.environ["OPENAI_API_KEY"] = "your_openai_key"
embeddings = OpenAIEmbeddings()

# Query to embed
query = "How are you?"
embedding = embeddings.embed_query(query)

# Display first 5 numbers
print("First 5 values of embedding:")
print(embedding[:5])
