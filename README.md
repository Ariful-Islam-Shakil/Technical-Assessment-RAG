# Retrieval-Augmented Generation (RAG) System

## Project Description

This project implements a **Retrieval-Augmented Generation (RAG) system** that can answer questions based on a collection of documents. The system combines semantic search with LLM-based answer generation to provide accurate and context-aware responses.

### Goals

- Build a document retrieval system using **FAISS** and embeddings.
- Chunk documents for efficient retrieval.
- Generate answers using **LLM (Groq API)** based on retrieved context.
- Support multiple document types (`.txt`, `.pdf`, Wikipedia articles).
- Provide a modular and extendable pipeline for experimentation.

### Project structure
```python
Technical-Assessment-RAG/
│
├── data/                  # Folder to store documents (.txt, .pdf)
├── src/
│   ├── document_loader.py # Load and chunk documents
│   ├── embeddings.py      # Load model and generate embeddings
│   ├── retriever.py       # FAISS index and retrieval
│   ├── llm_client.py      # Generate answers using Groq API
│   └── rag_pipeline.py    # Orchestrates building and querying RAG system
│
├── app.py                 # Streamlit interface
├── requirements.txt
└── README.md
```


### Features Workflow

1. **Document Collection**
   - Load `.txt` and `.pdf` files from the `data/` directory.
   - Optionally fetch Wikipedia articles for additional content.

2. **Preprocessing & Chunking**
   - Split documents into sentence-based chunks with configurable overlap.
   - Track metadata for each chunk (document title, chunk index).

3. **Embedding**
   - Load a sentence-transformers model from HuggingFace.
   - Generate embeddings for all chunks.

4. **Indexing**
   - Build a **FAISS index** for efficient similarity search.
   - Normalize embeddings for cosine similarity.

5. **Query Answering**
   - Embed the user query.
   - Retrieve top-k similar chunks from FAISS index.
   - Construct a context-aware prompt.
   - Generate answers using Groq LLM.

6. **Optional Enhancements**
   - Hybrid search (keyword + semantic).
   - Include source citations in generated answers.
   - Threshold-based filtering for relevance.

---

## Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Technical-Assessment-RAG.git
cd Technical-Assessment-RAG
```

### 2. Create Python Environment
```bash
# Windows
conda create -n .rag python=3.12.6
conda activate .rag
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Set Environment Variables

Create a `.env` file in the root directory and add your API key(s):
```ini
GROQ_API_KEY=your groq api key
LLM_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```
### 5. Run Streamlit App
```bash
streamlit run app.py
```