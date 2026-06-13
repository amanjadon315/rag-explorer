from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://raguser:ragpass@localhost:5432/ragexplorer"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_models_available: List[str] = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]

    # OpenAI (optional)
    openai_api_key: str = ""

    # FAISS
    faiss_index_path: str = "./data/faiss"

    # Chunking defaults
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64
    default_top_k: int = 5

    # App
    debug: bool = True
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
