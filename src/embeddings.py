import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load a sentence-transformers embedding model.

    Args:
        model_name (str): Name or path of the HuggingFace SentenceTransformer model.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of text strings using the given model.

    Args:
        model (SentenceTransformer): The embedding model.
        texts (List[str]): List of text strings to embed.

    Returns:
        np.ndarray: 2D numpy array where each row is the embedding of a text.
    """
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
