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

### 2. Install Dependencies
pip install -r requirements.txt

### 3. (Optional) Set Environment Variables
cp .env.example .env
#### Edit .env with your API keys

### 4. Index Your Documents
Place your documents (PDF, TXT, DOCX, MD) in the data/ folder.
python scripts/index_documents.py \
  --input_dir data/ \
  --chunk_size 512 \
  --chunk_overlap 50 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2

#### 5. Launch the Query Interface
Option A: Streamlit UI (Interactive)
```
streamlit run app.py
```
Option B: FastAPI Server
```
uvicorn api.server:app --reload --port 8000
```
Option C: CLI Query
```
python query.py --question "What are the main risks mentioned in the annual report?"
```
## 📁 Project Structure
```
├── README.md                  # You're here!
├── requirements.txt           # Python dependencies
├── config.yaml                # Pipeline settings (chunking, models, thresholds)
├── .env.example               # Template for API keys
├── data/                      # ← Put your documents here
├── index/                     # Stored vector database (FAISS/Chroma)
├── scripts/
│   ├── index_documents.py     # Ingestion + embedding pipeline
│   └── evaluate_rag.py        # Run benchmarks
├── app.py                     # Streamlit demo interface
├── api/
│   └── server.py              # FastAPI endpoints
├── notebooks/
│   └── rag_experiments.ipynb  # Exploratory analysis & testing
├── rag/
│   ├── loader.py              # Document loaders
│   ├── chunker.py             # Text splitting logic
│   ├── embedder.py            # Embedding wrapper
│   ├── retriever.py           # Search top-k chunks
│   └── generator.py           # LLM answer synthesis
└── utils/
    └── evaluation.py          # Metrics (BLEU, ROUGE, faithfulness)
```
## 💡 Usage Examples
Query via Python
```
from rag.pipeline import RAGPipeline

rag = RAGPipeline()
response = rag.query("Explain the revenue growth strategy in Q3.")
print(response.answer)
print(response.sources)  # List of document chunks used
```
Evaluate Performance
```
python scripts/evaluate_rag.py --test_data eval/qa_pairs.json
```
## 📊 Evaluation Metrics
```
Metric,Description
Retrieval Recall@K,% of ground-truth chunks in top-K
Answer Faithfulness,Does answer stay true to retrieved context?
Relevance Score,Semantic similarity to ideal answer
Latency,End-to-end query time
```
## 📊 Evaluation Metrics
<img width="600" height="311" alt="image" src="https://github.com/user-attachments/assets/baa6b9bb-7424-4df3-9557-d389e6b7a956" />

## 🤝 Contributing
We welcome contributions! See something missing?
1. Fork the repo
2. Create your feature branch (git checkout -b feature/amazing-rag)
3. Commit your changes (git commit -m 'Add amazing RAG feature')
4. Push and open a Pull Request

## 🚀 Future Roadmap
- [] Multi-modal RAG (images, tables, charts)
- [] Advanced reranking (Cross-Encoders, Cohere Rerank)
- [] Docker + Kubernetes deployment
- [] LangServe / Gradio UI options
- [] Persistent user sessions & chat history
- [] Integration with Notion, Google Drive, Confluence

## 📄 License
This project is licensed under the MIT License

## ⭐ Star this repo if you found it useful!
Built with ❤️ for the RAG community.
Have questions? Open an issue or reach out!

<ul>
  <li><span style="display:inline-block; width:20px; height:20px; border:2px solid #aaa; border-radius:5px; margin-right:8px;"></span>Multi-modal RAG (images, tables, charts)</li>
  <li><span style="display:inline-block; width:20px; height:20px; border:2px solid #aaa; border-radius:5px; margin-right:8px;"></span>Advanced reranking (Cross-Encoders, Cohere Rerank)</li>
</ul>


