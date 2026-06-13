from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.core.evaluation.evaluator import Evaluator

router = APIRouter()
evaluator = Evaluator()


class RetrievalEvalRequest(BaseModel):
    query: str
    retrieved_chunks: List[str]
    relevant_keywords: Optional[List[str]] = None


class HallucinationRequest(BaseModel):
    response: str
    context_chunks: List[str]


class RougeRequest(BaseModel):
    prediction: str
    reference: str


class FullEvalRequest(BaseModel):
    query: str
    retrieved_chunks: List[str]
    response: str
    reference_answer: Optional[str] = None


@router.post("/retrieval")
def eval_retrieval(body: RetrievalEvalRequest):
    metrics = evaluator.evaluate_retrieval(
        body.query, body.retrieved_chunks, body.relevant_keywords
    )
    return {
        "precision_at_k": metrics.precision_at_k,
        "recall_at_k": metrics.recall_at_k,
        "k": metrics.k,
        "n_relevant": metrics.n_relevant,
        "n_retrieved": metrics.n_retrieved,
        "n_relevant_retrieved": metrics.n_relevant_retrieved,
    }


@router.post("/hallucination")
def eval_hallucination(body: HallucinationRequest):
    result = evaluator.evaluate_hallucination(body.response, body.context_chunks)
    return {
        "overall_score": result.overall_score,
        "verdict": result.verdict,
        "sentences": result.sentences,
    }


@router.post("/rouge")
def eval_rouge(body: RougeRequest):
    result = evaluator.evaluate_rouge(body.prediction, body.reference)
    return {
        "rouge_1_f": result.rouge_1_f,
        "rouge_2_f": result.rouge_2_f,
        "rouge_l_f": result.rouge_l_f,
    }


@router.post("/full")
def full_evaluation(body: FullEvalRequest):
    """Run all evaluations in one call — the main evaluation endpoint."""
    retrieval = evaluator.evaluate_retrieval(body.query, body.retrieved_chunks)
    hallucination = evaluator.evaluate_hallucination(body.response, body.retrieved_chunks)
    rouge = evaluator.evaluate_rouge(body.response, body.reference_answer) if body.reference_answer else None

    return {
        "retrieval": {
            "precision_at_k": retrieval.precision_at_k,
            "recall_at_k": retrieval.recall_at_k,
            "k": retrieval.k,
        },
        "hallucination": {
            "overall_score": hallucination.overall_score,
            "verdict": hallucination.verdict,
            "sentences": hallucination.sentences,
        },
        "rouge": {
            "rouge_1_f": rouge.rouge_1_f,
            "rouge_2_f": rouge.rouge_2_f,
            "rouge_l_f": rouge.rouge_l_f,
        } if rouge else None,
    }
