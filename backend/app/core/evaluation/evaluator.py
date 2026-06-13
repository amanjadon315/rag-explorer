from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class RetrievalMetrics:
    precision_at_k: float      # fraction of retrieved chunks that are relevant
    recall_at_k: float         # fraction of relevant chunks that were retrieved
    k: int
    n_relevant: int
    n_retrieved: int
    n_relevant_retrieved: int


@dataclass
class HallucinationResult:
    response: str
    sentences: List[dict]      # [{sentence, supported, score}]
    overall_score: float       # 0=hallucinated, 1=fully grounded
    verdict: str               # "grounded" | "partial" | "hallucinated"


@dataclass
class RougeResult:
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float


@dataclass
class FullEvaluation:
    retrieval: Optional[RetrievalMetrics]
    hallucination: Optional[HallucinationResult]
    rouge: Optional[RougeResult]
    response_length_tokens: int


class Evaluator:
    """
    Evaluation suite for RAG pipelines.

    Retrieval evaluation:
      Precision@K = |relevant ∩ retrieved| / K
      Recall@K    = |relevant ∩ retrieved| / |relevant|

      "Relevant" is determined by keyword overlap with the query — a simple
      heuristic. Production systems use human-labeled ground truth or
      an LLM-as-judge approach.

    Hallucination detection:
      We split the response into sentences, then check each sentence for
      token overlap with the retrieved context. Low overlap = potential hallucination.
      (A proper implementation uses an NLI model like cross-encoder/nli-deberta.)

    ROUGE:
      ROUGE-L measures the longest common subsequence between reference and prediction.
      Useful when you have a reference answer.
    """

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: List[str],
        relevant_keywords: Optional[List[str]] = None,
    ) -> RetrievalMetrics:
        k = len(retrieved_chunks)

        # Use query words as relevance signal if no keywords provided
        keywords = relevant_keywords or [w.lower() for w in query.split() if len(w) > 3]

        def is_relevant(chunk: str) -> bool:
            chunk_lower = chunk.lower()
            return any(kw in chunk_lower for kw in keywords)

        n_relevant_retrieved = sum(1 for c in retrieved_chunks if is_relevant(c))
        # Estimate total relevant: assume corpus would have 2x what we retrieved (conservative)
        n_relevant_estimated = max(n_relevant_retrieved, len(keywords))

        precision = round(n_relevant_retrieved / k, 4) if k > 0 else 0.0
        recall = round(n_relevant_retrieved / n_relevant_estimated, 4) if n_relevant_estimated > 0 else 0.0

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            k=k,
            n_relevant=n_relevant_estimated,
            n_retrieved=k,
            n_relevant_retrieved=n_relevant_retrieved,
        )

    def evaluate_hallucination(
        self,
        response: str,
        context_chunks: List[str],
    ) -> HallucinationResult:
        import re
        # Split response into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
        context_text = " ".join(context_chunks).lower()
        context_words = set(context_text.split())

        results = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            # Remove stop words for a cleaner signal
            content_words = {w for w in sent_words if len(w) > 4}
            if not content_words:
                overlap = 1.0   # short sentences get benefit of the doubt
            else:
                overlap = len(content_words & context_words) / len(content_words)
            results.append({
                "sentence": sent,
                "overlap_score": round(overlap, 3),
                "supported": overlap >= 0.3,
            })

        n_supported = sum(1 for r in results if r["supported"])
        overall = round(n_supported / len(results), 3) if results else 1.0

        if overall >= 0.8:
            verdict = "grounded"
        elif overall >= 0.5:
            verdict = "partial"
        else:
            verdict = "hallucinated"

        return HallucinationResult(
            response=response,
            sentences=results,
            overall_score=overall,
            verdict=verdict,
        )

    def evaluate_rouge(self, prediction: str, reference: str) -> RougeResult:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return RougeResult(
            rouge_1_f=round(scores["rouge1"].fmeasure, 4),
            rouge_2_f=round(scores["rouge2"].fmeasure, 4),
            rouge_l_f=round(scores["rougeL"].fmeasure, 4),
        )
