# app.py
import streamlit as st
from backend import load_documents, create_vectorstore, build_qa_chain

st.set_page_config(page_title="RAG with Ollama", layout="wide")

st.title("📖 RAG Q&A App with Ollama")
st.write("Upload docs or provide a source, then ask questions!")

# Sidebar input
st.sidebar.header("Data Source")
source_type = st.sidebar.selectbox("Choose input type:", ["pdf", "txt_url", "web", "local"])
path_or_url = st.sidebar.text_input("Enter file path or URL:")

if st.sidebar.button("Load Documents"):
    with st.spinner("Loading and processing documents..."):
        docs, sources = load_documents(source_type, path_or_url)
        vectorstore = create_vectorstore(docs)
        qa_chain, retriever, llm = build_qa_chain(vectorstore)
        st.session_state.qa_chain = qa_chain
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.success(f"✅ Documents loaded from {path_or_url}")

# Question Answering
if "qa_chain" in st.session_state:
    query = st.text_input("Ask a question:")
    if query:
        result = st.session_state.qa_chain({"query": query})

        if result and result["result"]:
            st.subheader("📌 Answer")
            st.write(result["result"])

            st.subheader("🔗 Sources")
            for doc in result["source_documents"]:
                st.write(f"- {doc.metadata.get('source', 'N/A')}")
        else:
            st.warning("No relevant info found.")
