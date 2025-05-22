from langchain.text_splitter import RecursiveCharacterTextSplitter

latex_text = """
\\documentclass{article}
\\begin{document}
\\maketitle
\\section{Introduction}
Large language models (LLMs)... (text lanjutan)
\\end{document}
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

chunks = splitter.split_text(latex_text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
