"""
metric_lo.py — Lexical Overlap (ROUGE-L) metric (Zhang et al.).

Measures verbatim leakage of source document text into the RAG response.
Direction: ↓  (lower = better privacy)
No LLM required.

Formulas (exact match to paper):
  Precision = LCS(response, doc) / len(response)
  Recall    = LCS(response, doc) / len(doc)
  F1        = 2·P·R / (P + R)
"""
from rouge_score import rouge_scorer


def compute_lo(response: str, source_doc_text: str) -> dict[str, float]:
    """Compute ROUGE-L between a single response and its source document."""
    if not response.strip() or not source_doc_text.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # rouge_score convention: score(target, prediction)
    # target = source doc, prediction = RAG response
    scores = scorer.score(source_doc_text, response)
    rL = scores["rougeL"]

    return {
        "precision": rL.precision,
        "recall": rL.recall,
        "f1": rL.fmeasure,
    }


def aggregate_lo(results: list[dict[str, float]]) -> dict[str, float]:
    """Mean P / R / F1 over N instances."""
    if not results:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = len(results)
    return {
        "precision": sum(r["precision"] for r in results) / n,
        "recall": sum(r["recall"] for r in results) / n,
        "f1": sum(r["f1"] for r in results) / n,
    }
