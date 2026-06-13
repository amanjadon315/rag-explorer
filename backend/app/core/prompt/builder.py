from dataclasses import dataclass, field
from typing import List, Optional
from app.core.ingestion.tokenizer import Tokenizer


SYSTEM_TEMPLATES = {
    "rag_default": (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context does not contain the answer, say so clearly. "
        "Do not make up information."
    ),
    "rag_analytical": (
        "You are an analytical assistant. Use the provided context to answer the question. "
        "Cite specific parts of the context when relevant. "
        "If the context is insufficient, explain what information is missing."
    ),
    "rag_concise": (
        "Answer the question in 2-3 sentences using only the provided context. Be direct."
    ),
    "open_ended": (
        "You are a helpful assistant. Answer the user's question to the best of your ability."
    ),
}


@dataclass
class PromptSection:
    name: str            # "system" | "context" | "query"
    content: str
    token_count: int
    char_count: int


@dataclass
class BuiltPrompt:
    sections: List[PromptSection]
    final_prompt: str            # the full assembled string
    messages: List[dict]         # chat-format messages for OpenAI/Ollama API
    total_tokens: int
    context_window: int          # model's max context
    utilization_pct: float       # total_tokens / context_window * 100
    system_template: str
    n_context_chunks: int


class PromptBuilder:
    """
    Assembles the final prompt from:
      1. A system prompt (sets the LLM's behavior)
      2. Retrieved context chunks (injected as grounding)
      3. The user's query

    Exposing this stage lets you see exactly what the model receives —
    critical for debugging hallucinations and context utilization.
    """

    CONTEXT_WINDOW = 8192     # llama3.2 default; override per model

    def __init__(self, context_window: int = 8192):
        self.context_window = context_window
        self._tokenizer = Tokenizer()

    def build(
        self,
        query: str,
        context_chunks: List[str],
        system_template: str = "rag_default",
        custom_system_prompt: Optional[str] = None,
        chunk_separator: str = "\n\n---\n\n",
    ) -> BuiltPrompt:
        system_text = custom_system_prompt or SYSTEM_TEMPLATES.get(
            system_template, SYSTEM_TEMPLATES["rag_default"]
        )

        # Assemble context block
        if context_chunks:
            context_text = (
                "CONTEXT:\n"
                + chunk_separator.join(
                    f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
                )
            )
        else:
            context_text = ""

        # Build sections with token counts
        sections = []
        for name, content in [("system", system_text), ("context", context_text), ("query", query)]:
            if content:
                tc = self._tokenizer.count_tokens(content)
                sections.append(PromptSection(
                    name=name,
                    content=content,
                    token_count=tc,
                    char_count=len(content),
                ))

        # Final assembled prompt (for display)
        final_prompt = f"{system_text}\n\n{context_text}\n\nQUESTION: {query}" if context_text else f"{system_text}\n\nQUESTION: {query}"

        # Chat messages format (for API)
        messages = [{"role": "system", "content": system_text}]
        if context_text:
            messages.append({"role": "user", "content": f"{context_text}\n\nQUESTION: {query}"})
        else:
            messages.append({"role": "user", "content": query})

        total_tokens = sum(s.token_count for s in sections)
        utilization = round(total_tokens / self.context_window * 100, 1)

        return BuiltPrompt(
            sections=sections,
            final_prompt=final_prompt,
            messages=messages,
            total_tokens=total_tokens,
            context_window=self.context_window,
            utilization_pct=utilization,
            system_template=system_template,
            n_context_chunks=len(context_chunks),
        )

    def get_templates(self) -> dict:
        return SYSTEM_TEMPLATES
