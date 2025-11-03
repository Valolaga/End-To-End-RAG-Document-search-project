# End-to-End RAG Document Search Project

![RAG Pipeline](https://img.shields.io/badge/RAG-Pipeline-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

A **complete, modular, and production-ready** Retrieval-Augmented Generation (RAG) system for intelligent document search and question answering. This project enables you to ingest documents, build a searchable vector index, and generate accurate, context-grounded answers using state-of-the-art LLMs — all in one end-to-end pipeline.

> **Reduce LLM hallucinations. Boost answer relevance. Scale to thousands of documents.**

---

## 🚀 Key Features

- **Full RAG Pipeline**: Document loading → Chunking → Embedding → Indexing → Retrieval → Generation
- **Flexible Embeddings**: Supports OpenAI, Hugging Face (Sentence Transformers), or local models
- **Vector Stores**: FAISS (local), Chroma, or Pinecone (cloud-ready)
- **Smart Chunking**: Configurable chunk size, overlap, and metadata extraction
- **Hybrid Search Ready**: Combine semantic + keyword (BM25) search
- **Evaluation Suite**: Measure retrieval recall, answer faithfulness, and relevance
- **Interactive UI**: Streamlit app for live querying
- **API Endpoint**: FastAPI server for integration into apps
- **Extensible Design**: Easy to plug in new LLMs, retrievers, or document types

---

## 🛠 Tech Stack

| Component           | Technology |
|---------------------|----------|
| Framework           | LangChain / LlamaIndex |
| Embeddings          | `sentence-transformers`, OpenAI |
| Vector DB           | FAISS, Chroma, Pinecone |
| LLM                 | OpenAI GPT, Llama 3 (via Ollama), Hugging Face |
| UI                  | Streamlit |
| API                 | FastAPI + Uvicorn |
| Language            | Python 3.8+ |

---

## 📋 Prerequisites

- Python 3.8 or higher
- `pip` or `conda` for package management
- API keys (optional):
  - `OPENAI_API_KEY` for GPT models
  - `PINECONE_API_KEY` for cloud vector DB
- GPU recommended for local LLMs/embeddings

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Valolaga/End-To-End-RAG-Document-search-project.git
cd End-To-End-RAG-Document-search-project
```

2. Install Dependencies
pip install -r requirements.txt

3. (Optional) Set Environment Variables
cp .env.example .env
#### Edit .env with your API keys

4. Index Your Documents
Place your documents (PDF, TXT, DOCX, MD) in the data/ folder.
python scripts/index_documents.py \
  --input_dir data/ \
  --chunk_size 512 \
  --chunk_overlap 50 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2





