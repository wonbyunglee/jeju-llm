# Hot Place Recommendation Chatbot in Jeju Ireland using LLM & RAG

![alt text](image.png)

* Award in 2024 Bigcontest Generative-AI Session

## Introduction

This chatbot is a service that recommends famous restaurants in Jeju Island. Based on the data provided by Shinhan Card, we classify hot places and cool places and recommend them. When users ask questions that fit the situation, we recommend famous restaurants and hidden restaurants according to the context. We implemented this service by using the RAG technique and LLM.

## Technology Stack

- Data Source: Shinhan Card sales data

- RAG (Retrieval-Augmented Generation)
    - Keyword-based search: BM25 (rank_bm25 library)
    - Semantic similarity search: FAISS (Facebook AI Similarity Search)

- LLM (Large Language Model)
    - Google Gemini API (gemini-1.5-flash)

- Web Service Deployment
    - Streamlit-based UI
    - Torch & Hugging Face Transformers for embedding generation

## How It Works

1. Data Processing & Embedding Generation
- chunking.py: Processes Shinhan Card data and generates summarized texts.
- embeddings.py: Uses bge-m3-korean model to build FAISS index.

2. Search & Recommendation System
- BM25 Keyword Search: Uses Kiwi morphological analyzer to filter relevant documents.
- FAISS Semantic Search: Retrieves the most relevant documents from BM25 results.

3. LLM-Based Response Generation
- Combines BM25 + FAISS results and generates a natural language response using Gemini API.
- Categorizes recommendations into hot places (trendy restaurants) and cool places (hidden gems).