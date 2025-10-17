import faiss
import numpy as np
from typing import Tuple
from loguru import logger 

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index using inner product (cosine similarity) for a set of embeddings.

    Args:
        embeddings (np.ndarray): 2D array of shape (num_texts, embedding_dim) representing
                                 the embeddings of your corpus. Should be float32.
    Returns:
        faiss.Index: A FAISS IndexFlatIP index containing the normalized embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def retrieve(index: faiss.Index, query_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-k most similar embeddings from the FAISS index for a given query embedding.

    Args:
        index (faiss.Index): FAISS index containing corpus embeddings.
        query_emb (np.ndarray): 2D array of shape (1, embedding_dim) for a single query embedding.
        top_k (int): Number of top results to return.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - scores (np.ndarray): Array of similarity scores of top-k results.
            - indices (np.ndarray): Array of indices of top-k results in the original corpus.
    """
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, top_k)
    return scores[0], indices[0]
