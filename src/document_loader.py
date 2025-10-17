import os
import wikipedia 
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
from loguru import logger 
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

################################
# Fetch Wikipedia Articles
################################
def fetch_wikipedia_articles(topics, save_dir="data") -> None:
    """
    Fetch multiple Wikipedia articles and save as .txt files.

    Args:
        topics (list): List of article topics.
        save_dir (str): Directory to store text files.
    """
    os.makedirs(save_dir, exist_ok=True)

    for topic in topics:
        try:
            print(f"Fetching: {topic} ...")
            page = wikipedia.page(topic)
            content = page.content

            # Clean content (optional: remove references)
            clean_content = content.replace("== References ==", "").strip()

            file_path = os.path.join(save_dir, f"{topic.replace(' ', '_')}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_content)
            print(f"Saved: {file_path}")

        except Exception as e:
            logger.error(f"Failed to fetch {topic}: {e}")

###############################################
# Load documents from .txt and .pdf files
###############################################

def load_documents(directory: str = "data") -> List[Dict[str, str]]:
    """
    Load all .txt and .pdf documents from a directory.

    Args:
        directory (str): Path to folder containing the documents.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with 'title' and 'content'.
    """
    documents = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip directories
        if not os.path.isfile(file_path):
            continue

        content = ""
        try:
            # Handle text files
            if filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            # Handle PDF files
            elif filename.lower().endswith(".pdf"):
                reader = PdfReader(file_path)
                print(f"Number of pages: {len(reader.pages)}")
                text = ""
                page_no = 0
                for page in reader.pages:
                    text += page.extract_text() or ""
                    page_no += 1
                    if page_no % 10 == 0:
                        logger.info(f"Loaded {page_no} pages from {filename}...")
                content = text

            # Skip empty or unreadable files
            if content.strip():
                documents.append({
                    "title": os.path.splitext(filename)[0],
                    "content": content.strip()
                })
                print(f"Loaded: {filename}")
            else:
                print(f"Skipped empty file: {filename}")

        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents

###############################
# Chunk text
############################### 

def chunk_text(text: str, chunk_size: int = 5, overlap: int = 1) -> List[str]:
    """
    Split text into sentence-based chunks with overlap using NLTK sentence tokenizer.
    
    Args:
        text (str): The input text.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of overlapping sentences between chunks.
    
    Returns:
        List[str]: List of text chunks.
    """
    
    # Tokenize single file text into sentences 
    sentences = sent_tokenize(text.strip()) 
    chunks = []
    start = 0
    
    # Chunking with overlap based on given number of sentences
    while start < len(sentences):
        end = min(start + chunk_size, len(sentences))
        chunk = " ".join(sentences[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  
    return chunks

###########################
# Build corpus with metadata
###########################

def build_corpus(docs: List[Dict[str, str]], chunk_size: int = 5, overlap: int = 2)-> Tuple[List[str], List[Dict[str, int]]]:
    """
    Build a corpus of text chunks from documents and track metadata.

    Args:
        docs (List[Dict[str, str]]): List of documents, each with 'title' and 'content'.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of overlapping sentences between chunks.

    Returns:
        Tuple[List[str], List[Dict[str, int]]]:
            - texts: List of chunked text strings
            - metadata: List of dictionaries containing metadata for each chunk:
                - 'title': document title
                - 'chunk_index': index of chunk in document
    """
    texts, metadata = [], []
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size, overlap)
        for i, c in enumerate(chunks):
            texts.append(c)
            metadata.append({
                "title": doc.get("title", ""),
                "chunk_index": i
            })
        print(f"Processsed {doc.get('title')} -> {len(chunks)}")
    return texts, metadata
