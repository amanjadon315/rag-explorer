from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.core.prompt.builder import PromptBuilder

router = APIRouter()
builder = PromptBuilder()


class BuildRequest(BaseModel):
    query: str
    context_chunks: List[str] = []
    system_template: str = "rag_default"
    custom_system_prompt: Optional[str] = None


@router.get("/templates")
def get_templates():
    return {"templates": builder.get_templates()}


@router.post("/build")
def build_prompt(body: BuildRequest):
    result = builder.build(
        query=body.query,
        context_chunks=body.context_chunks,
        system_template=body.system_template,
        custom_system_prompt=body.custom_system_prompt,
    )
    return {
        "final_prompt": result.final_prompt,
        "messages": result.messages,
        "total_tokens": result.total_tokens,
        "context_window": result.context_window,
        "utilization_pct": result.utilization_pct,
        "n_context_chunks": result.n_context_chunks,
        "sections": [
            {
                "name": s.name,
                "content": s.content,
                "token_count": s.token_count,
                "char_count": s.char_count,
            }
            for s in result.sections
        ],
    }
