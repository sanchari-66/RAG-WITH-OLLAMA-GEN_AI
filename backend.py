# backend.py
import os
import requests
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Wrapper for docs
class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

SOURCES_DICTIONARY = {}

def load_documents(choice, path_or_url):
    documents = []
    if choice == "pdf":
        loader = PyMuPDFLoader(path_or_url)
        docs = loader.load()
        for d in docs:
            documents.append(Document(metadata={"source": path_or_url}, page_content=d.page_content))
        SOURCES_DICTIONARY["pdf"] = path_or_url

    elif choice == "txt_url":
        response = requests.get(path_or_url)
        if response.status_code == 200:
            documents.append(Document(metadata={"source": path_or_url}, page_content=response.text))
            SOURCES_DICTIONARY["txt_url"] = path_or_url

    elif choice == "web":
        loader = WebBaseLoader(path_or_url)
        docs = loader.load()
        for d in docs:
            documents.append(Document(metadata={"source": path_or_url}, page_content=d.page_content))
        SOURCES_DICTIONARY["webpage"] = path_or_url

    elif choice == "local":
        if path_or_url.endswith(".pdf"):
            loader = PyMuPDFLoader(path_or_url)
        else:
            loader = TextLoader(path_or_url, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            documents.append(Document(metadata={"source": path_or_url}, page_content=d.page_content))
        SOURCES_DICTIONARY["local_file"] = path_or_url

    return documents, SOURCES_DICTIONARY


def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    split_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(metadata=doc.metadata, page_content=chunk))

    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_store"
    )
    return vectorstore


# def build_qa_chain(vectorstore):
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#     llm = Ollama(model="mistral")

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )
#     return qa_chain, retriever, llm

def build_qa_chain(vectorstore):
    # Use similarity score threshold so unrelated queries yield 0 docs
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.2}  # tune threshold if needed
    )

    llm = Ollama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain, retriever, llm

REFUSAL_PHRASES = (
    "the provided context does not",
    "the given context does not",
    "no information in the context",
    "the context doesn't provide",
    "the context does not provide",
    "based on the provided context",
    "not mentioned in the context",
    "i do not have enough information from the context",
)

def _looks_unhelpful(answer: str) -> bool:
    a = (answer or "").lower()
    if len(a.strip()) < 15:
        return True
    return any(p in a for p in REFUSAL_PHRASES)

def answer_query(query, qa_chain=None, retriever=None, llm=None):
    """
    Try to answer from documents first; if retrieval fails or the answer looks
    context-refusal/unhelpful, fall back to general LLM.
    Returns: (answer, sources_list)
    """
    # If we don't have the pieces for RAG, just LLM it.
    if not (retriever and qa_chain and llm):
        # Safety: still allow pure LLM
        llm = llm or Ollama(model="mistral")
        response = llm.invoke(query)
        return response, ["None (general knowledge)"]

    # 1) Retrieve relevant docs
    retrieved_docs = retriever.get_relevant_documents(query)

    # 2) If nothing relevant, go straight to LLM
    if not retrieved_docs or not any(getattr(d, "page_content", "").strip() for d in retrieved_docs):
        response = llm.invoke(query)
        return response, ["None (general knowledge)"]

    # 3) Ask with RAG
    result = qa_chain({"query": query})
    answer = result.get("result", "")
    sources = [d.metadata.get("source", "N/A") for d in result.get("source_documents", [])]

    # 4) If the RAG answer looks like a "no info in context" message, fall back
    if _looks_unhelpful(answer):
        response = llm.invoke(query)
        return response, ["None (general knowledge)"]

    # 5) Otherwise return the doc-grounded answer
    return answer, sources
