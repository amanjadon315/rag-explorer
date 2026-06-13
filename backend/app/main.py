from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from app.config import get_settings
from app.api.v1 import ingestion, embeddings, retrieval, prompt, inference, evaluation
from app.utils.database import init_db

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB tables, warm up embedding model."""
    logger.info("Starting RAG Explorer API")
    await init_db()
    # Warm up embedding model (loads weights into memory once)
    from app.dependencies import get_embedder
    embedder = get_embedder()
    _ = embedder.embed(["warm up"])
    logger.info("Embedding model ready", model=settings.embedding_model)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Explorer API",
    description="Interactive LLM & RAG Architecture Explorer — inspect every stage of the pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Ingestion"])
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["Embeddings"])
app.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["Retrieval"])
app.include_router(prompt.router, prefix="/api/v1/prompt", tags=["Prompt"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["Inference"])
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["Evaluation"])


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.embedding_model, "llm": settings.llm_model}
