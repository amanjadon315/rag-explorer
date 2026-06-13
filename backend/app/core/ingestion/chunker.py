import re
from dataclasses import dataclass, field
from typing import List, Literal
from enum import Enum


ChunkStrategy = Literal["fixed", "recursive", "sentence"]


@dataclass
class Chunk:
    index: int
    text: str
    start_char: int
    end_char: int
    token_estimate: int       # rough estimate: len(text.split()) * 1.3
    strategy: str
    chunk_size: int
    overlap: int


@dataclass
class ChunkResult:
    chunks: List[Chunk]
    strategy: str
    chunk_size: int
    overlap: int
    total_chunks: int
    avg_chunk_length: float
    avg_token_estimate: float


class Chunker:
    """
    Three chunking strategies — each surfaces different tradeoffs:

    - fixed:      Split every N characters. Fast, predictable, ignores semantics.
                  Risk: cuts mid-sentence, degrading retrieval quality.

    - recursive:  Try splitting on paragraphs → sentences → words in order.
                  Preserves natural boundaries as much as possible.
                  This is what LangChain's RecursiveCharacterTextSplitter does.

    - sentence:   Split on sentence boundaries (periods, ?, !).
                  Then greedily pack sentences into chunks up to size limit.
                  Best semantic coherence; slowest to compute.
    """

    def chunk(
        self,
        text: str,
        strategy: ChunkStrategy = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> ChunkResult:
        if strategy == "fixed":
            chunks = self._fixed(text, chunk_size, overlap)
        elif strategy == "recursive":
            chunks = self._recursive(text, chunk_size, overlap)
        elif strategy == "sentence":
            chunks = self._sentence(text, chunk_size, overlap)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        avg_len = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
        avg_tok = sum(c.token_estimate for c in chunks) / len(chunks) if chunks else 0

        return ChunkResult(
            chunks=chunks,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            total_chunks=len(chunks),
            avg_chunk_length=round(avg_len, 1),
            avg_token_estimate=round(avg_tok, 1),
        )

    # ------------------------------------------------------------------ #
    # Fixed: slice every chunk_size characters, step back by overlap      #
    # ------------------------------------------------------------------ #
    def _fixed(self, text: str, size: int, overlap: int) -> List[Chunk]:
        chunks = []
        step = size - overlap
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end]
            chunks.append(self._make_chunk(idx, chunk_text, start, end, "fixed", size, overlap))
            idx += 1
            start += step
        return chunks

    # ------------------------------------------------------------------ #
    # Recursive: split on \n\n → \n → ". " → " " until size fits         #
    # ------------------------------------------------------------------ #
    def _recursive(self, text: str, size: int, overlap: int) -> List[Chunk]:
        separators = ["\n\n", "\n", ". ", " ", ""]
        raw = self._recursive_split(text, separators, size)
        return self._merge_with_overlap(raw, size, overlap, "recursive")

    def _recursive_split(self, text: str, separators: List[str], size: int) -> List[str]:
        if len(text) <= size:
            return [text]
        sep = separators[0] if separators else ""
        parts = text.split(sep) if sep else list(text)
        result = []
        for part in parts:
            if len(part) > size and len(separators) > 1:
                result.extend(self._recursive_split(part, separators[1:], size))
            else:
                result.append(part)
        return [p for p in result if p.strip()]

    # ------------------------------------------------------------------ #
    # Sentence: split on sentence boundaries then pack greedily           #
    # ------------------------------------------------------------------ #
    def _sentence(self, text: str, size: int, overlap: int) -> List[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        raw = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= size:
                current = (current + " " + sent).strip()
            else:
                if current:
                    raw.append(current)
                current = sent
        if current:
            raw.append(current)
        return self._merge_with_overlap(raw, size, overlap, "sentence")

    # ------------------------------------------------------------------ #
    # Shared: stitch raw splits back into overlap-aware Chunk objects     #
    # ------------------------------------------------------------------ #
    def _merge_with_overlap(
        self, parts: List[str], size: int, overlap: int, strategy: str
    ) -> List[Chunk]:
        chunks = []
        start_char = 0
        for i, part in enumerate(parts):
            end_char = start_char + len(part)
            chunks.append(self._make_chunk(i, part, start_char, end_char, strategy, size, overlap))
            start_char = end_char + 1
        return chunks

    def _make_chunk(
        self, idx: int, text: str, start: int, end: int,
        strategy: str, size: int, overlap: int
    ) -> Chunk:
        # Rough token estimate: word count × 1.3 (accounts for subword tokenization)
        token_estimate = int(len(text.split()) * 1.3)
        return Chunk(
            index=idx,
            text=text,
            start_char=start,
            end_char=end,
            token_estimate=token_estimate,
            strategy=strategy,
            chunk_size=size,
            overlap=overlap,
        )
