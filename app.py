import os
import streamlit as st
from loguru import logger
from src.document_loader import load_documents, fetch_wikipedia_articles
from src.rag_pipeline import build_rag_system, answer_query 

# -------------------------------
# 🎨 Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="RAG Demo with Groq", layout="wide")

st.title("🔍 Retrieval-Augmented Generation (RAG) Demo")
st.markdown("Use your own or Wikipedia documents to answer questions with **Groq LLM**.")

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# -------------------------------
# ⚙️ Sidebar Configuration
# -------------------------------
st.sidebar.header("⚙️ RAG Configuration")

# --- 🧹 Clean Data Folder ---
if st.sidebar.button("Clean Data Folder"):
    try:
        for file in os.listdir("data"):
            os.remove(os.path.join("data", file))
        st.sidebar.success("✅ All files deleted from /data")
    except Exception as e:
        st.sidebar.error(f"❌ Error cleaning folder: {e}")

# --- 📁 Upload Documents ---
st.sidebar.subheader("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload text or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.sidebar.success(f"✅ Uploaded {len(uploaded_files)} file(s) saved to /data")

# --- 🌐 Fetch Wikipedia Articles ---
st.sidebar.subheader("🌐 Fetch Wikipedia Articles")
wiki_topics = st.sidebar.text_area(
    "Enter topics (comma-separated):",
    placeholder="e.g. Artificial Intelligence, Quantum Computing, Machine Learning"
)
fetch_wiki = st.sidebar.button("📘 Fetch & Save Wikipedia Articles")

if fetch_wiki and wiki_topics.strip():
    topics = [t.strip() for t in wiki_topics.split(",") if t.strip()]
    with st.spinner("Fetching Wikipedia articles..."):
        try:
            fetch_wikipedia_articles(topics, save_dir="data")
            st.sidebar.success(f"✅ Fetched {len(topics)} article(s) saved to /data")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to fetch articles: {e}")
            logger.error(f"Wikipedia fetch failed: {e}")

# --- 🔧 Model Configuration ---
embed_model = st.sidebar.selectbox(
    "Select Embedding Model",
    options=[
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "thenlper/gte-small",
        "intfloat/e5-small-v2",
        "BAAI/bge-small-en"
    ],
    index=0
)

chunk_size = st.sidebar.slider("Chunk Size", 5, 20, 10, 5)
overlap = st.sidebar.slider("Overlap", 1, int(chunk_size - 1), 2, 1)
top_k = st.sidebar.slider("Top K", 1, 10, 3, 1)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05)

# --- 🚀 Build RAG System ---
build_clicked = st.sidebar.button("🚀 Build RAG System")

# -------------------------------
# 🧩 Build RAG System
# -------------------------------
if build_clicked:
    with st.spinner("Building RAG System... Please wait."):
        logger.info("User clicked to build RAG system.")
        try:
            docs = load_documents("data")
            if len(docs) == 0:
                st.error("❌ No documents found in /data folder.")
                st.stop()

            index, texts, metadata, model = build_rag_system(docs, chunk_size, overlap, embed_model)

            st.session_state["rag_ready"] = True
            st.session_state["index"] = index 
            st.session_state["texts"] = texts
            st.session_state["metadata"] = metadata
            st.session_state["model"] = model

            st.success("✅ RAG System built successfully! You can now query below.")
        except Exception as e:
            st.error(f"❌ Error building RAG system: {e}")
            logger.error(f"Build failed: {e}")

# -------------------------------
# 💬 Query Interface
# -------------------------------
if "rag_ready" in st.session_state and st.session_state["rag_ready"]:
    st.markdown("---")
    st.subheader("💬 Ask a Question")

    user_query = st.text_area("Enter your query:", placeholder="e.g. What are transformer models?")
    query_button = st.button("🔎 Get Answer")

    if query_button and user_query.strip():
        with st.spinner("Generating answer..."):
            result = answer_query(
                st.session_state["index"], 
                st.session_state["texts"],
                st.session_state["metadata"],
                st.session_state["model"],
                user_query,
                top_k,
                similarity_threshold
            )

            st.write("### 🧠 Answer:")
            st.markdown(result["answer"])

            with st.expander("📄 Retrieved Contexts"):
                for i, r in enumerate(result["retrieved"], 1):
                    st.markdown(f"**{i}. {r['metadata']['title']}** — (Score: {r['score']:.4f})")
                    st.write(r["text"])
else:
    st.warning("👈 Upload or fetch documents, configure settings, and click '🚀 Build RAG System' first.")
