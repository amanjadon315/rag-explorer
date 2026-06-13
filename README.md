# Interactive LLM & RAG Architecture Explorer

A portfolio-grade educational platform that exposes every internal stage of a RAG pipeline — from document ingestion through LLM inference and evaluation. Built to demonstrate deep understanding of AI systems, not just usage of them.

---

## What this project demonstrates

| Concept | Where it shows up |
|---|---|
| Tokenization internals | `/api/v1/ingestion/tokenize` — BPE token IDs, subword breakdown |
| Dense vector embeddings | `core/embeddings/embedder.py` — sentence-transformers, L2 norms |
| FAISS HNSW indexing | `core/vectorstore/faiss_store.py` — index construction, efSearch |
| BM25 sparse retrieval | `core/vectorstore/bm25_store.py` — TF-IDF, term saturation |
| Reciprocal Rank Fusion | `core/retrieval/hybrid_retriever.py` — score-free fusion |
| Prompt engineering | `core/prompt/builder.py` — context injection, token budgeting |
| Streaming inference | `core/llm/ollama_client.py` — SSE token-by-token streaming |
| RAG evaluation | `core/evaluation/evaluator.py` — Precision@K, hallucination scoring |
| Dimensionality reduction | `core/embeddings/reducer.py` — PCA explained variance, t-SNE |

---

## Tech stack (all free, all local)

- **Backend**: Python 3.11 · FastAPI · SQLAlchemy
- **LLM**: Ollama (llama3.2 or any local model)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) — no API key
- **Vector store**: FAISS HNSW + BM25 (rank-bm25)
- **Frontend**: React 18 · TypeScript · Vite · Tailwind · Plotly.js
- **Infrastructure**: PostgreSQL · Redis · Docker Compose

---

## Prerequisites

Install these once:

```bash
# 1. Python 3.11+
python --version

# 2. Node 18+
node --version

# 3. Docker + Docker Compose
docker --version
docker compose version

# 4. Ollama (local LLM runtime — free)
# macOS:
brew install ollama
# Linux:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: download from https://ollama.com

# 5. Pull a model (one-time, ~2GB)
ollama pull llama3.2
```

---

## Setup — Option A: Docker Compose (recommended)

```bash
# Clone / enter the project
cd rag-explorer

# Copy environment file
cp .env.example .env

# Start everything (Postgres + Redis + backend + frontend)
docker compose up --build

# Backend:  http://localhost:8000
# Frontend: http://localhost:5173
# API docs: http://localhost:8000/docs
```

The first build takes ~5 minutes — it downloads the embedding model weights inside the container.

---

## Setup — Option B: Local dev (faster iteration)

### Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit env
cp ../.env.example .env
# Edit DATABASE_URL if you're running Postgres locally, or use SQLite:
# DATABASE_URL=sqlite+aiosqlite:///./rag_explorer.db

# Create data directory for FAISS
mkdir -p data/faiss

# Run the API
uvicorn app.main:app --reload --port 8000
```

> **No Postgres?** Replace `asyncpg` with `aiosqlite` and set:
> `DATABASE_URL=sqlite+aiosqlite:///./rag_explorer.db`
> Add `aiosqlite` to requirements.txt.

### Frontend

```bash
cd frontend

npm install
npm run dev
# Runs at http://localhost:5173
```

### Start Ollama

```bash
# In a separate terminal
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

---

## Using the app

### Full pipeline walkthrough (Pipeline page)

1. **Upload** a document (txt, md, pdf) or paste text
2. **Clean** — see what noise was removed
3. **Chunk** — adjust strategy (recursive/fixed/sentence), size, and overlap. Watch chunks appear.
4. **Embed + Index** — embeds all chunks into FAISS + BM25 in one click
5. **Search** — type a query, run hybrid retrieval, see ranked results with RRF scores
6. **Build prompt** — see system prompt + context + query assembled with token counts
7. **Run inference** — tokens stream in real time from Ollama
8. **Evaluate** — Precision@K, hallucination score, per-sentence grounding

### Embedding Explorer

- Add your own sentences
- Visualize them in 2D (PCA or t-SNE) — semantically similar sentences cluster together
- Compute cosine similarity between any two texts
- Build a pairwise similarity heatmap

### Retrieval Explorer

- Paste a knowledge base document
- Run the same query through **vector search**, **BM25**, and **hybrid RRF** simultaneously
- Compare scores side by side — notice when keyword queries favor BM25 and semantic queries favor vector

### Prompt Inspector

- See system prompt, retrieved context, and user query in separate panels with per-section token counts
- Edit the system prompt and watch the token budget update
- Inspect the raw messages array sent to the LLM

### Inference Dashboard

- Adjust temperature and max_tokens with sliders
- Toggle between streaming (SSE) and non-streaming mode
- See tokens/second throughput

### Evaluation Panel

- Input any query + retrieved chunks + model response
- Get Precision@K, Recall@K, sentence-level hallucination breakdown, and ROUGE scores
- Works with demo data — no prior pipeline run needed

---

## API documentation

FastAPI generates interactive docs automatically:

```
http://localhost:8000/docs       # Swagger UI
http://localhost:8000/redoc      # ReDoc
```

Key endpoints:

```
POST /api/v1/ingestion/upload        Upload file, extract text
POST /api/v1/ingestion/chunk         Chunk text, return all chunks + metadata
POST /api/v1/ingestion/tokenize      Tokenize text, show token IDs + subwords
POST /api/v1/embeddings/generate     Embed texts, return vectors
POST /api/v1/embeddings/similarity   Cosine similarity between two texts
POST /api/v1/embeddings/reduce       PCA/t-SNE reduction for visualization
POST /api/v1/retrieval/index         Embed chunks and store in FAISS + BM25
POST /api/v1/retrieval/vector        Vector similarity search
POST /api/v1/retrieval/bm25          BM25 sparse search
POST /api/v1/retrieval/hybrid        Hybrid RRF search
POST /api/v1/prompt/build            Assemble final prompt with token counts
POST /api/v1/inference/complete      Non-streaming LLM completion
POST /api/v1/inference/stream        SSE streaming completion
POST /api/v1/evaluation/full         Full evaluation suite
```

---

## Project structure

```
rag-explorer/
├── backend/
│   └── app/
│       ├── main.py                  FastAPI app, lifespan, router registration
│       ├── config.py                Settings from .env
│       ├── dependencies.py          Singleton embedder + vector stores
│       ├── api/v1/                  Route handlers (thin — delegate to core)
│       │   ├── ingestion.py
│       │   ├── embeddings.py
│       │   ├── retrieval.py
│       │   ├── prompt.py
│       │   ├── inference.py
│       │   └── evaluation.py
│       └── core/                    All business logic lives here
│           ├── ingestion/           loader, cleaner, chunker, tokenizer
│           ├── embeddings/          embedder, reducer (PCA/t-SNE)
│           ├── vectorstore/         faiss_store, bm25_store
│           ├── retrieval/           hybrid_retriever (RRF)
│           ├── prompt/              builder, templates
│           ├── llm/                 ollama_client (streaming + non-streaming)
│           └── evaluation/          evaluator (retrieval, hallucination, rouge)
└── frontend/
    └── src/
        ├── pages/                   One page per pipeline stage
        │   ├── PipelineView.tsx     Master end-to-end pipeline
        │   ├── EmbeddingExplorer.tsx
        │   ├── RetrievalExplorer.tsx
        │   ├── PromptInspector.tsx
        │   ├── InferenceDashboard.tsx
        │   └── EvaluationPanel.tsx
        ├── components/ui/           MetricBadge, ScoreBar, TokenBar, etc.
        ├── store/pipelineStore.ts   Zustand global state
        └── lib/api.ts               Typed API client
```

---

## Adding OpenAI (optional)

To compare Ollama with GPT-4o, add to `.env`:
```
OPENAI_API_KEY=sk-...
```

Then create `backend/app/core/llm/openai_client.py`:
```python
from openai import AsyncOpenAI
client = AsyncOpenAI()

async def complete(messages, temperature=0.7, max_tokens=1024):
    resp = await client.chat.completions.create(
        model="gpt-4o", messages=messages,
        temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content
```

The prompt builder and inference router are designed to swap clients with one line.

---

## Common issues

**`Ollama not reachable`** — run `ollama serve` in a terminal. In Docker, set `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

**`FAISS dimension mismatch`** — happens if you switch embedding models after indexing. Delete `backend/data/faiss/` and re-index.

**`torch slow on CPU`** — normal for sentence-transformers on CPU. First embed call loads model weights. Subsequent calls are fast. For GPU: install `torch` with CUDA and sentence-transformers will use it automatically.

**`Out of memory during t-SNE`** — reduce number of sentences or use PCA instead.

---

## Extending the project

Ideas to add next:
- **Cross-encoder reranking**: add `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank top-k results
- **Semantic chunking**: use embeddings to find natural topic boundaries in text
- **Multi-document sessions**: store document metadata in Postgres, support multiple knowledge bases
- **RAGAS integration**: `pip install ragas` for faithfulness, answer relevancy, context precision
- **Streaming evaluation**: evaluate hallucination token by token as the response generates
