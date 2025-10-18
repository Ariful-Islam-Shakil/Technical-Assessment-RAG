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
   - Three `text` file and one `pdf` file are already included
   - Optionally fetch Wikipedia articles or can share `txt` or `pdf` file for additional content.

2. **Preprocessing & Chunking**
   - Split documents into `sentence-based` chunks with configurable overlap.
   - Track metadata for each chunk (document title, chunk index).

3. **Embedding**
   - Load a `sentence-transformers` model from `HuggingFace`.
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
   - Include source citations in generated answers.
   - Threshold-based filtering for relevance.

---

## Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Ariful-Islam-Shakil/Technical-Assessment-RAG.git
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

## Some qureie
| #  | Query                                                                 | Actual Answer                                                                                     | RAG Response       |
|----|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------|------------------|
| 1  | What is supervised learning in AI?                                   | A type of machine learning where models are trained on labeled data to predict outputs.   | Supervised learning in AI is a type of machine learning where the model is trained using labeled data, where input features and corresponding outputs are known. This approach enables the model to learn from examples and make predictions on new, unseen data. Common supervised algorithms include linear regression, logistic regression, decision trees, support vector machines (SVM), and neural networks [AI_ML] contents:.  |
| 2  | Name two deep learning models for image and text data.               | CNN for images, Transformers for text.                                                    | For image data, one popular deep learning model is Convolutional Neural Networks (CNNs). For text data, a popular deep learning model is Transformers [AI_ML] contents.  |
| 3  | What is the fatality rate of COVID-19 as of November 1, 2020?        | Approximately 2.6%.                                                                       | The estimated case fatality rate of COVID-19 as of November 1, 2020, was 2.60% [covid19].  |
| 4  | Which protein in SARS-CoV-2 helps the virus bind to host cells?      | Spike (S) glycoprotein.                                                                  | The protein in SARS-CoV-2 that helps the virus bind to host cells is the Spike (S) protein, specifically the Receptor-Binding Domain (RBD) of the S protein. This RBD binds to the human Angiotensin-Converting Enzyme 2 (ACE2) receptor on host cells [covid19].  |
| 5  | What is a P/E ratio in stock market analysis?                         | Price-to-Earnings ratio, calculated as Market Price per Share divided by EPS.            | The Price-to-Earnings (P/E) ratio is calculated by dividing the market price per share by earnings per share (EPS), helping investors understand if a stock is overvalued or undervalued [stock_market_faundamental]. |
| 6  | Name one mRNA vaccine developed for COVID-19.                         | Moderna mRNA-1273.                                                                        | One mRNA vaccine developed for COVID-19 is the vaccine developed by Moderna [covid19]. Additionally, BioNTech, Germany, also developed an mRNA vaccine against COVID-19 [covid19].  |
| 7  | What is the main goal of reinforcement learning?                      | To train agents to maximize cumulative reward by interacting with an environment.        | The main goal of reinforcement learning is to maximize cumulative reward over time [AI_ML].Reinforcement learning trains agents to make sequential decisions by interacting with an environment, and the agent receives rewards or penalties based on its actions [AI_ML].The goal is achieved through key concepts including state, action, reward, policy, and value function [AI_ML].  |
| 8  | What are bull and bear markets?                                       | Bull markets have rising stock prices; bear markets have falling stock prices.           | A bull market is characterized by rising prices and optimism, while a bear market is defined by falling prices and pessimism. This occurs as the stock market moves in cycles influenced by economic conditions, investor sentiment, and global events [stock_market_faundamental].  |
| 9  | How many candidate vaccines for COVID-19 were in development globally by October 19, 2020? | 212 candidate vaccines.                                                                   | According to the information provided in [covid19], as of October 19, 2020, there were 212 SARS-CoV-2 candidate vaccines being developed globally. Additionally, another source [covid19] also mentions that researchers around the world are developing 213 COVID-19 candidate vaccines. However, the more specific and earlier date reference is 212 candidate vaccines [covid19].  |
| 10 | Where is Dhaka located? | No relevant documents found.                   | No relevant documents found.  |
