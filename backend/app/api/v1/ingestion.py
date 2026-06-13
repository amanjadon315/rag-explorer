from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.core.ingestion.loader import DocumentLoader
from app.core.ingestion.cleaner import TextCleaner
from app.core.ingestion.chunker import Chunker, ChunkStrategy
from app.core.ingestion.tokenizer import Tokenizer

router = APIRouter()
loader = DocumentLoader()
cleaner = TextCleaner()
chunker = Chunker()
tokenizer = Tokenizer()


class TextInput(BaseModel):
    text: str
    filename: str = "paste.txt"


class ChunkRequest(BaseModel):
    text: str
    strategy: ChunkStrategy = "recursive"
    chunk_size: int = 512
    overlap: int = 64


class TokenizeRequest(BaseModel):
    text: str
    encoding: str = "cl100k_base"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and extract raw text."""
    content = await file.read()
    try:
        result = loader.load_bytes(content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "filename": result.filename,
        "file_type": result.file_type,
        "char_count": result.char_count,
        "word_count": result.word_count,
        "preview": result.text[:500],
        "text": result.text,
    }


@router.post("/paste")
async def paste_text(body: TextInput):
    """Accept pasted text directly."""
    result = loader.load_text(body.text, body.filename)
    return {
        "filename": result.filename,
        "file_type": result.file_type,
        "char_count": result.char_count,
        "word_count": result.word_count,
        "text": result.text,
    }


@router.post("/clean")
async def clean_text(body: TextInput):
    """Clean text and show what changed."""
    result = cleaner.clean(body.text)
    return {
        "original": result.original,
        "cleaned": result.cleaned,
        "changes": result.changes,
        "original_length": result.original_length,
        "cleaned_length": result.cleaned_length,
        "reduction_pct": result.reduction_pct,
    }


@router.post("/chunk")
async def chunk_text(body: ChunkRequest):
    """Chunk text and return all chunks with metadata."""
    result = chunker.chunk(
        body.text,
        strategy=body.strategy,
        chunk_size=body.chunk_size,
        overlap=body.overlap,
    )
    return {
        "strategy": result.strategy,
        "chunk_size": result.chunk_size,
        "overlap": result.overlap,
        "total_chunks": result.total_chunks,
        "avg_chunk_length": result.avg_chunk_length,
        "avg_token_estimate": result.avg_token_estimate,
        "chunks": [
            {
                "index": c.index,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "token_estimate": c.token_estimate,
            }
            for c in result.chunks
        ],
    }


@router.post("/tokenize")
async def tokenize_text(body: TokenizeRequest):
    """Tokenize text and expose token IDs and subword breakdown."""
    tok = Tokenizer(body.encoding)
    result = tok.tokenize(body.text)
    return {
        "text": result.text,
        "token_count": result.token_count,
        "vocabulary_size": result.vocabulary_size,
        "encoding_name": result.encoding_name,
        "token_details": result.token_details,
    }
