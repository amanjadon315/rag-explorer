import httpx
import json
import time
from dataclasses import dataclass
from typing import List, AsyncGenerator, Optional
from app.config import get_settings

settings = get_settings()


@dataclass
class CompletionResult:
    response: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    tokens_per_second: float
    finish_reason: str


class OllamaClient:
    """
    Calls Ollama's REST API — runs models locally (llama3.2, mistral, etc.)
    No API key. No cost. Runs on your machine.

    Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md

    The /api/chat endpoint accepts messages in OpenAI format so this client
    can be swapped for the OpenAI client with minimal changes.
    """

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.llm_model
        self.timeout = httpx.Timeout(120.0)  # LLMs can be slow locally

    async def complete(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> CompletionResult:
        """Non-streaming completion — waits for full response."""
        t0 = time.perf_counter()
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        response_text = data["message"]["content"]
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        tokens_per_sec = round(eval_count / (eval_duration_ns / 1e9), 1) if eval_duration_ns else 0

        return CompletionResult(
            response=response_text,
            model=model or self.model,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_sec,
            finish_reason=data.get("done_reason", "stop"),
        )

    async def stream(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion — yields tokens as they arrive.
        Used with FastAPI's StreamingResponse / Server-Sent Events.
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    async def list_models(self) -> List[str]:
        """List models available in the local Ollama installation."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
