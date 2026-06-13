# Interactive LLM & RAG Architecture Explorer

> An educational platform that exposes every internal stage of a RAG pipeline — from document ingestion through LLM inference and evaluation. Built to demonstrate deep understanding of AI systems architecture.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat&logo=fastapi)
![React](https://img.shields.io/badge/React-18-blue?style=flat&logo=react)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

---

## What this project demonstrates

This is not a chatbot. It's an **AI systems inspector** — every stage of the pipeline is visible, adjustable, and explained.

| Concept | Implementation |
|---|---|
| Tokenization internals | BPE token IDs, subword breakdown via tiktoken |
| Dense vector embeddings | sentence-transformers, L2 norms, cosine similarity |
| FAISS HNSW indexing | Approximate nearest-neighbor, efSearch tuning |
| BM25 sparse retrieval | TF-IDF scoring, term saturation |
| Reciprocal Rank Fusion | Score-free hybrid search combining vector + BM25 |
| Prompt engineering | Context injection, token budget tracking |
| Streaming inference | Server-Sent Events, token-by-token output |
| RAG evaluation | Precision@K, Recall@K, hallucination detection, ROUGE |
| Dimensionality reduction | PCA explained variance, t-SNE clustering |

---

## Architecture
Document Input

↓

Text Cleaning → tracks every normalization applied

↓

Chunking → fixed / recursive / sentence strategies

↓

Tokenization → BPE subwords, token IDs

↓

Embedding Generation → sentence-transformers (local, no API key)

↓

Vector Storage → FAISS HNSW + BM25 sparse index

↓

Similarity Search → cosine similarity, top-k retrieval

↓

Hybrid RRF → Reciprocal Rank Fusion combining both retrievers

↓

Prompt Construction → system + context + query with token counts

↓

LLM Inference → Ollama (local) with SSE token streaming

↓

Evaluation → Precision@K, hallucination scoring, ROUGE
---

## Pages

### Pipeline View
Walk through all 11 stages end-to-end. Every stage shows its input, output, and key metrics in real time.

### Embedding Explorer
- Visualize semantic space in 2D (PCA or t-SNE)
- Compute cosine similarity between any two texts
- Build pairwise similarity heatmaps
- See why semantically similar sentences cluster together

### Retrieval Explorer
- Run the same query through vector search, BM25, and hybrid RRF simultaneously
- Side-by-side score comparison with bar charts
- See when keyword queries favor BM25 vs semantic queries favor vector search

### Prompt Inspector
- Three-panel view: system prompt · retrieved context · user query
- Live token count per section
- Context window utilization bar
- Inspect the exact messages array sent to the LLM

### Inference Dashboard
- Token streaming via Server-Sent Events
- Adjustable temperature and max_tokens
- Tokens/second throughput metric
- Context window utilization visualization

### Evaluation Panel
- Precision@K and Recall@K with gauge visualization
- Sentence-level hallucination detection with ✓/✗ per sentence
- ROUGE-1, ROUGE-2, ROUGE-L scores with bar charts

---

##  Tech Stack

**Backend**
- Python 3.11 · FastAPI · SQLAlchemy + SQLite
- sentence-transformers (all-MiniLM-L6-v2) — runs fully locally
- FAISS HNSW for vector search
- rank-bm25 for sparse retrieval
- Ollama for local LLM inference (llama3.2:1b)
- tiktoken for tokenization
- rouge-score for evaluation

**Frontend**
- React 18 · TypeScript · Vite
- Tailwind CSS · JetBrains Mono
- Plotly.js for scatter plots and heatmaps
- Recharts for metric visualizations
- Zustand for state management

---

##  Running locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed

### Setup

```bash
# 1. Clone
git clone https://github.com/amanjadon315/rag-explorer.git
cd rag-explorer

# 2. Pull LLM model (one-time, ~800MB)
ollama pull llama3.2:1b

# 3. Backend
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
mkdir data\faiss

# 4. Configure environment
copy .env.example .env
# Edit .env — set DATABASE_URL=sqlite+aiosqlite:///./rag_explorer.db

# 5. Start backend
uvicorn app.main:app --reload --port 8000

# 6. Frontend (new terminal)
cd frontend
npm install
npm run dev

# 7. Ollama (new terminal)
ollama serve
```

Open **http://localhost:5173**

---

##  Project Structure
rag-explorer/

├── backend/

│   └── app/

│       ├── api/v1/          # Route handlers — ingestion, embeddings, retrieval, prompt, inference, evaluation

│       ├── core/

│       │   ├── ingestion/   # loader, cleaner, chunker, tokenizer

│       │   ├── embeddings/  # embedder (sentence-transformers), reducer (PCA/t-SNE)

│       │   ├── vectorstore/ # FAISS HNSW index, BM25 store

│       │   ├── retrieval/   # hybrid RRF retriever

│       │   ├── prompt/      # prompt builder, system templates

│       │   ├── llm/         # Ollama client (streaming + non-streaming)

│       │   └── evaluation/  # Precision@K, hallucination, ROUGE

│       ├── main.py          # FastAPI app entry point

│       ├── config.py        # Settings from .env

│       └── dependencies.py  # Singleton embedder + vector stores

└── frontend/

└── src/

├── pages/           # One page per pipeline stage

├── components/ui/   # MetricBadge, ScoreBar, TokenBar, StreamingText

├── store/           # Zustand global pipeline state

└── lib/api.ts       # Typed API client
---

## Key design decisions

**Why FAISS HNSW over flat index?** HNSW gives O(log n) query time vs O(n) for brute force. For a portfolio demo the difference is small, but the architecture scales to millions of vectors.

**Why Reciprocal Rank Fusion?** RRF doesn't require score normalization — vector cosine scores and BM25 raw scores are incomparable. RRF fuses rankings instead of scores, which is both simpler and more robust.

**Why sentence-transformers locally?** Zero API cost, no rate limits, works offline, and demonstrates that production-quality embeddings don't require external services.

**Why expose intermediate outputs?** The goal is explainability. Every stage shows its input, output, and metrics so you can see exactly why the model answered the way it did.

---

##  Interview questions this project answers

- Walk me through the end-to-end RAG pipeline you built
- How does hybrid retrieval work? What is RRF and why use it over score averaging?
- What embedding model did you use and why? How did you compare alternatives?
- How does HNSW differ from a flat FAISS index?
- How did you detect hallucinations in LLM outputs?
- How does context window utilization affect RAG quality?
- What chunking strategy works best and why?
- How would you scale this to millions of documents?

---

## 📄 License

MIT