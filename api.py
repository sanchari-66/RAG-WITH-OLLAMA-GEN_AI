# api.py
from flask import Flask, request, jsonify
from backend import load_documents, create_vectorstore, build_qa_chain, answer_query

app = Flask(__name__)

# Root route to avoid 404
@app.route("/")
def home():
    return "RAG Ollama API is running!"


# Global objects
qa_chain = None
retriever = None
llm = None

@app.route("/load_documents", methods=["POST"])
def load_docs():
    global qa_chain, retriever, llm

    data = request.json
    source_type = data.get("source_type")
    path_or_url = data.get("path_or_url")

    documents, sources = load_documents(source_type, path_or_url)
    vectorstore = create_vectorstore(documents)
    qa_chain, retriever, llm = build_qa_chain(vectorstore)

    return jsonify({"status": "Documents loaded", "sources": sources})

@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain, retriever, llm

    if not qa_chain:
        return jsonify({"error": "No documents loaded yet!"}), 400

    query = request.json.get("query")
    answer, sources = answer_query(query, qa_chain, retriever, llm)
    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
