import numpy as np
from typing import List, Dict, Any, Tuple
from loguru import logger 
from .document_loader import build_corpus
from .embeddings import load_embedding_model, embed_texts
from .retriever import build_faiss_index, retrieve
from .llm_client import generate_answer


# -------------------------------
# Stage 1: Build RAG Components
# -------------------------------
def build_rag_system(
    docs: List[Dict[str, str]],
    CHUNK_SIZE: int,
    OVERLAP: int,
    EMBED_MODEL_NAME: str
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]], Any]:
    """
    Builds the RAG system components including corpus embeddings, FAISS index, and metadata.

    Args:
        docs (List[Dict[str, str]]): List of documents, each containing 'title' and 'content'.
        CHUNK_SIZE (int): Number of sentences per chunk.
        OVERLAP (int): Number of overlapping sentences between chunks.
        EMBED_MODEL_NAME (str): HuggingFace sentence-transformers model name for embeddings.

    Returns:
        Tuple[np.ndarray, List[str], List[Dict[str, Any]], Any]:
            - index: FAISS index containing corpus embeddings.
            - texts: List of text chunks.
            - metadata: List of metadata dictionaries for each chunk.
            - model: Loaded embedding model.
    """
    logger.info("Building RAG system...")

    # Build text corpus and metadata
    texts, metadata = build_corpus(docs, CHUNK_SIZE, OVERLAP)
    logger.info(f"Total chunks created: {len(texts)}")

    # Load embedding model
    model = load_embedding_model(EMBED_MODEL_NAME)
    logger.info("Embedding model loaded successfully")

    # Create embeddings for corpus
    corpus_emb = embed_texts(model, texts)
    logger.info("Document embeddings generated")

    # Build FAISS index
    index = build_faiss_index(np.copy(corpus_emb))
    logger.info("FAISS index built successfully")

    return index, texts, metadata, model


# -------------------------------
# Stage 2: Query Answering
# -------------------------------
def answer_query(
    index: np.ndarray,
    texts: List[str],
    metadata: List[Dict[str, Any]],
    model: Any,
    user_query: str,
    TOP_K: int,
    SIMILARITY_THRESHOLD: float
) -> Dict[str, Any]:
    """
    Retrieve relevant context and generate an answer for a single query using the RAG system.

    Args:
        index (np.ndarray): FAISS index containing corpus embeddings.
        texts (List[str]): List of text chunks.
        metadata (List[Dict[str, Any]]): List of metadata for each chunk.
        model (Any): Embedding model instance.
        user_query (str): Query string from user.
        TOP_K (int): Number of top chunks to retrieve.
        SIMILARITY_THRESHOLD (float): Minimum similarity score to consider a chunk relevant.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'answer': Generated answer string.
            - 'retrieved': List of retrieved chunks with metadata and scores.
            - 'prompt': Full prompt sent to the LLM.
    """
    logger.info(f"Processing user query: {user_query}")

    # Embed the query
    query_emb = embed_texts(model, [user_query])
    logger.info("Query embedded successfully")

    # Retrieve similar chunks from FAISS
    scores, indices = retrieve(index, query_emb, TOP_K)
    logger.info(f"Retrieved top-{TOP_K} similar chunks from index")

    retrieved: List[Dict[str, Any]] = []
    for idx, score in zip(indices, scores):
        if 0 <= idx < len(texts):
            retrieved.append({
                "metadata": metadata[idx],
                "text": texts[idx],
                "score": float(score)
            })

    # Handle no relevant documents
    if not retrieved or retrieved[0]["score"] < SIMILARITY_THRESHOLD:
        logger.warning("No relevant documents found for the query.")
        return {"answer": "No relevant documents found.", "retrieved": retrieved, "prompt": ""}

    # Prepare context and LLM prompt
    context = "\n\n".join([f"\nTitle:\n[{r['metadata']['title']}] contents:\n{r['text']}" for r in retrieved])
    prompt = (
        f"Based on the given contexts, answer the question or query. "
        f"Also include references using the [title] format.\n\n"
        f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    )
    logger.info("Prompt prepared for LLM")

    # Generate answer using Groq LLM
    try:
        answer = generate_answer(prompt)
        logger.info("Answer generated successfully using Groq API")
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        answer = f"Groq API error: {e}\n\nFallback context: \n{context[:500]}"

    return {"answer": answer, "retrieved": retrieved, "prompt": prompt}
