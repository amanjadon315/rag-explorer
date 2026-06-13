from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from app.core.llm.ollama_client import OllamaClient

router = APIRouter()
client = OllamaClient()


class CompletionRequest(BaseModel):
    messages: List[dict]
    temperature: float = 0.7
    max_tokens: int = 1024
    model: Optional[str] = None


@router.get("/models")
async def list_models():
    """List models available in local Ollama."""
    try:
        models = await client.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(503, f"Ollama not reachable: {e}. Is Ollama running?")


@router.get("/health")
async def ollama_health():
    ok = await client.health_check()
    return {"ollama_running": ok}


@router.post("/complete")
async def complete(body: CompletionRequest):
    """Non-streaming completion — returns full response with metrics."""
    try:
        result = await client.complete(
            messages=body.messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            model=body.model,
        )
    except Exception as e:
        raise HTTPException(503, f"LLM inference failed: {e}")

    return {
        "response": result.response,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": result.latency_ms,
        "tokens_per_second": result.tokens_per_second,
        "finish_reason": result.finish_reason,
    }


@router.post("/stream")
async def stream_completion(body: CompletionRequest):
    """
    Streaming completion via Server-Sent Events.
    Frontend reads this with EventSource or fetch + ReadableStream.
    Each event is a plain text token.
    """
    async def event_generator():
        try:
            async for token in client.stream(
                messages=body.messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                model=body.model,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
