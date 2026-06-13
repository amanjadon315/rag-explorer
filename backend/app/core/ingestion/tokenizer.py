from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TokenizeResult:
    text: str
    tokens: List[str]           # decoded token strings e.g. ["Hello", "Ġworld"]
    token_ids: List[int]        # raw integer IDs
    token_count: int
    vocabulary_size: int
    encoding_name: str
    # Per-token detail for visualization
    token_details: List[Dict]   # [{id, token, bytes, is_special}]


class Tokenizer:
    """
    Wraps tiktoken (GPT-style BPE tokenizer) to expose internals.

    Why this matters for RAG:
    - Token count determines if a chunk fits in the context window.
    - Different models use different tokenizers — same text = different token counts.
    - Subword tokenization explains why "tokenization" might become ["token", "ization"].
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)

    def tokenize(self, text: str) -> TokenizeResult:
        token_ids = self.enc.encode(text)
        tokens = [self.enc.decode([tid]) for tid in token_ids]

        token_details = []
        for tid, tok in zip(token_ids, tokens):
            token_details.append({
                "id": tid,
                "token": repr(tok),           # shows \n, spaces as visible chars
                "display": tok.replace(" ", "·"),  # middot for spaces — great for UI
                "byte_length": len(tok.encode("utf-8")),
                "is_special": tid >= 100000,   # tiktoken special token range
            })

        return TokenizeResult(
            text=text,
            tokens=tokens,
            token_ids=token_ids,
            token_count=len(token_ids),
            vocabulary_size=self.enc.n_vocab,
            encoding_name=self.encoding_name,
            token_details=token_details,
        )

    def count_tokens(self, text: str) -> int:
        """Fast path — just the count, no details."""
        return len(self.enc.encode(text))

    def count_tokens_for_messages(self, messages: List[Dict]) -> int:
        """Count tokens for a list of chat messages (includes per-message overhead)."""
        total = 0
        for msg in messages:
            total += 4  # per-message overhead
            for key, value in msg.items():
                total += self.count_tokens(str(value))
        total += 2  # reply priming
        return total
