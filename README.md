# RAG-WITH-OLLAMA-GEN_AI
# 📖 RAG Q&A App with Ollama

This repository contains a **Retrieval-Augmented Generation (RAG) Q&A system** using **Ollama** and **LangChain**.  
It allows users to upload documents (PDF, TXT, webpage) and ask questions based on the content.  
If the documents do not provide the answer, it falls back to **Ollama LLM** for general knowledge.

---

## 🚀 Features

- Load documents from:
  - PDF files
  - TXT files (local or via URL)
  - Webpage URLs
  - Local file paths
- Automatic text chunking with metadata preservation
- Vector search using Ollama embeddings
- Retrieval-based Q&A with fallback to Ollama LLM
- Streamlit frontend for interactive question-answering
- Support for Jupyter notebook exploration

---

## 🗂 Repository Structure
```
RAG_5/
│
├─ app.py # Streamlit app frontend
├─ backend.py # Core functions: load documents, create vectorstore, QA chain
├─ rag_with_ollama_user_input.ipynb # Notebook for experimentation
├─ requirement.txt # Dependencies
└─ .gitignore # Files to ignore in Git

```
---

## ⚡ Requirements

- Python 3.10+
- Packages:
  ```bash
  pip install -r requirement.txt

  🛠 Setup

Clone the repository:
```
git clone https://github.com/sanchari-66/RAG-WITH-OLLAMA-GEN_AI.git
cd RAG-WITH-OLLAMA-GEN_AI/RAG_5
```

Install dependencies:
```
pip install -r requirement.txt
```

Ensure Ollama is running and models (e.g., mistral) are available.

💻 Running Locally (Streamlit)

Start the Streamlit app:

streamlit run app.py


Sidebar: Upload your document or provide a source URL.

Ask questions in the text box.
If the context doesn’t have the answer, Ollama LLM provides a fallback response.


# 📝 Backend Workflow

## Load Documents

load_documents(choice, path_or_url)
Loads PDFs, TXT files, webpages, or local files.

## Vector Store Creation

create_vectorstore(documents)
Splits text into chunks and creates a Chroma vector store using Ollama embeddings.

## Build QA Chain

build_qa_chain(vectorstore)
Creates a RetrievalQA chain with Ollama LLM.

## Answer Queries

answer_query(query, qa_chain, retriever, llm)
Retrieves relevant documents. If no answer is found, uses Ollama general knowledge.


# ⚡ Future Improvements

Automation to regularly ingest new documents from URLs or folders.

Add authentication for multi-user support.

Deploy on a cloud platform for public access.
