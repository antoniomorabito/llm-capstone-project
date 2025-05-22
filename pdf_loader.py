from langchain.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("documents/A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf")
documents = loader.load()

# Display first 1000 characters
print(documents[0].page_content[:1000])
