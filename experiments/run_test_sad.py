# experiments/run_test_sad.py
"""
MLflow experiment "test_sad" — évalue le SAD Detector (CPB Block 6)
sur toutes les queries du benchmark avec Llama + CPBNaiveRAG.

Ce que on logue par query :
  - Métriques CPB standard (pii_leakage, query_risk, chunk decisions…)
  - Métriques SAD : sad_detected, decision, categories, confidence,
    max_similarity, filter_triggered
  - Ground truth : sensitivity (label du dataset), pour évaluer
    precision/recall du détecteur

Un run final "summary" logue les métriques agrégées.

Usage :
    python experiments/run_test_sad.py
    python experiments/run_test_sad.py --n-queries 20   # test rapide
"""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from config import MLFLOW_TRACKING_URI, QUERY_LOG_MAX_CHARS, TOP_K
from countermeasure.cpb_naive_rag import CPBNaiveRAG
from countermeasure.sad_detector import DEFAULT_SBERT_THRESHOLD
from data.query_generator import load_queries
from llms.llama_llm import LlamaLLM
from metrics.pii_leakage import compute_pii_leakage
from rag.naive_rag import NaiveRAG
from vectorstore.chroma_store import ChromaStore

EXPERIMENT_NAME = "test_sad_v2"
LLM_NAME = "llama3.1:8b"
RAG_NAME = "cpb_naive_rag"
SBERT_MODE = "f2_non_blocking"   # v4: F2 informational only, Phi-3 always runs when F1 passes

# Encode string decision → int for MLflow metric
_DECISION_CODE = {"pass": 0, "mask": 1, "reask": 2, "block": 3}

SENSITIVE_LABELS = {"HEALTH", "POLITICS", "ETHNIC", "SEX", "BELIEF"}


def _safe(value, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _cpb_chunk_counts(chunk_decisions: list) -> tuple[int, int]:
    n_masked = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "mask")
    n_suppressed = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "suppress")
    return n_masked, n_suppressed


def _is_true_positive(sad_detected: bool, ground_truth_sensitivity: str | None) -> int:
    """SAD détecté ET le dataset dit que la query est sensible."""
    return int(sad_detected and ground_truth_sensitivity in SENSITIVE_LABELS)


def _is_false_positive(sad_detected: bool, ground_truth_sensitivity: str | None) -> int:
    """SAD détecté MAIS le dataset dit que la query n'est PAS sensible."""
    return int(sad_detected and ground_truth_sensitivity not in SENSITIVE_LABELS)


def _is_false_negative(sad_detected: bool, ground_truth_sensitivity: str | None) -> int:
    """SAD non détecté MAIS le dataset dit que la query est sensible."""
    return int(not sad_detected and ground_truth_sensitivity in SENSITIVE_LABELS)


def log_query_run(query_item: dict, result: dict) -> None:
    raw_chunks = result.get("raw_chunks", [])
    masked_chunks = result.get("chunks", [])
    sad = result.get("cpb_sad_result")
    ground_truth = query_item.get("sensitivity") or ""

    pii = compute_pii_leakage(response=result["response"], chunks=raw_chunks, query=query_item["query"])
    audit = result.get("cpb_audit")
    chunk_decisions = result.get("cpb_chunk_decisions", [])
    n_masked_chunks, n_suppressed_chunks = _cpb_chunk_counts(chunk_decisions)

    sad_detected = result.get("cpb_sad_detected", False)
    sad_decision = result.get("cpb_sad_decision", "pass")
    sad_categories = result.get("cpb_sad_categories", [])
    sad_confidence = result.get("cpb_sad_confidence", 0.0)
    sad_max_sim = sad.max_similarity if sad else 0.0
    sad_filter = result.get("cpb_sad_filter", 0)

    run_name = f"llama__{query_item.get('query_id', 'unknown')}"
    with mlflow.start_run(run_name=run_name):
        # ── Params ────────────────────────────────────────────────────────
        mlflow.log_param("llm", LLM_NAME)
        mlflow.log_param("rag_architecture", RAG_NAME)
        mlflow.log_param("query_id", query_item.get("query_id", "unknown"))
        mlflow.log_param("query_type", query_item.get("query_type", "unknown"))
        mlflow.log_param("entity_type", query_item.get("entity_type") or "")
        mlflow.log_param("doc_id", query_item.get("doc_id") or "")
        mlflow.log_param("query_preview", (result.get("cpb_masked_query") or query_item["query"])[:QUERY_LOG_MAX_CHARS])
        mlflow.log_param("response_preview", result["response"][:QUERY_LOG_MAX_CHARS])
        mlflow.log_param("cpb_response_guard_decision", result.get("cpb_response_guard_decision", ""))

        # Ground truth label du dataset — clé pour évaluer la détection
        mlflow.log_param("ground_truth_sensitivity", ground_truth)

        # SAD params
        mlflow.log_param("sad_decision", sad_decision)
        mlflow.log_param("sad_categories", ",".join(sad_categories) if sad_categories else "none")
        mlflow.log_param("sad_sbert_threshold", DEFAULT_SBERT_THRESHOLD)
        mlflow.log_param("sad_sbert_mode", SBERT_MODE)
        mlflow.log_param("sad_reasoning", (sad.reasoning if sad else "")[:200])

        # ── Metrics standard ──────────────────────────────────────────────
        mlflow.log_metric("tokens_prompt", _safe(result.get("tokens_prompt")))
        mlflow.log_metric("tokens_completion", _safe(result.get("tokens_completion")))
        mlflow.log_metric("tokens_total", _safe(result.get("tokens_total")))

        mlflow.log_metric("pii_leakage_rate", pii.leakage_rate)
        mlflow.log_metric("n_pii_total", pii.n_pii_total)
        mlflow.log_metric("n_pii_leaked", pii.n_pii_leaked)

        mlflow.log_metric("n_raw_chunks", len(raw_chunks))
        mlflow.log_metric("n_masked_chunks", len(masked_chunks))
        mlflow.log_metric("cpb_n_masked_chunks", n_masked_chunks)
        mlflow.log_metric("cpb_n_suppressed_chunks", n_suppressed_chunks)
        mlflow.log_metric("cpb_query_risk", _safe(result.get("cpb_query_risk")))
        mlflow.log_metric("cpb_query_pii_score", _safe(result.get("cpb_query_pii_score")))

        if audit is not None:
            mlflow.log_metric("cpb_max_pii_score", audit.max_pii_score)
            mlflow.log_metric("cpb_min_budget", audit.min_budget)
            mlflow.log_metric("cpb_response_leakage_score", audit.leakage_score)

        # ── Métriques SAD ─────────────────────────────────────────────────
        mlflow.log_metric("sad_detected", int(sad_detected))
        mlflow.log_metric("sad_decision_code", _DECISION_CODE.get(sad_decision, 0))
        mlflow.log_metric("sad_confidence", _safe(sad_confidence))
        mlflow.log_metric("sad_max_similarity", _safe(sad_max_sim))
        mlflow.log_metric("sad_filter_triggered", _safe(sad_filter))
        mlflow.log_metric("sad_n_categories", len(sad_categories))

        # Métriques d'évaluation (precision/recall)
        mlflow.log_metric("sad_true_positive", _is_true_positive(sad_detected, ground_truth))
        mlflow.log_metric("sad_false_positive", _is_false_positive(sad_detected, ground_truth))
        mlflow.log_metric("sad_false_negative", _is_false_negative(sad_detected, ground_truth))
        # 1 si la catégorie détectée correspond au label ground truth
        category_match = int(any(c == ground_truth for c in sad_categories))
        mlflow.log_metric("sad_category_match", category_match)


def log_summary_run(stats: dict, n_queries: int) -> None:
    """Log un run d'agrégat avec les métriques globales de l'expérience."""
    with mlflow.start_run(run_name="__summary__"):
        mlflow.log_param("llm", LLM_NAME)
        mlflow.log_param("rag_architecture", RAG_NAME)
        mlflow.log_param("sad_sbert_threshold", DEFAULT_SBERT_THRESHOLD)
        mlflow.log_param("sad_sbert_mode", SBERT_MODE)
        mlflow.log_param("n_queries_total", n_queries)

        n_detected = stats["n_detected"]
        n_tp = stats["n_tp"]
        n_fp = stats["n_fp"]
        n_fn = stats["n_fn"]
        n_sensitive = stats["n_sensitive"]  # queries with a sensitive label

        detection_rate = n_detected / n_queries if n_queries else 0.0
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        mlflow.log_metric("n_queries", n_queries)
        mlflow.log_metric("n_sad_detected", n_detected)
        mlflow.log_metric("sad_detection_rate", detection_rate)
        mlflow.log_metric("sad_precision", precision)
        mlflow.log_metric("sad_recall", recall)
        mlflow.log_metric("sad_f1", f1)
        mlflow.log_metric("n_true_positive", n_tp)
        mlflow.log_metric("n_false_positive", n_fp)
        mlflow.log_metric("n_false_negative", n_fn)
        mlflow.log_metric("n_queries_sensitive", n_sensitive)

        # Distribution des décisions
        for decision, count in stats["decision_counts"].items():
            mlflow.log_metric(f"n_decision_{decision}", count)

        # Distribution des catégories SAD détectées
        for cat, count in stats["category_counts"].items():
            mlflow.log_metric(f"n_sad_category_{cat}", count)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Queries         : {n_queries}")
        print(f"  SAD detected    : {n_detected} ({detection_rate:.1%})")
        print(f"  Precision       : {precision:.3f}")
        print(f"  Recall          : {recall:.3f}")
        print(f"  F1              : {f1:.3f}")
        print(f"  TP / FP / FN    : {n_tp} / {n_fp} / {n_fn}")
        print(f"  Decisions       : {dict(stats['decision_counts'])}")
        print(f"  Categories      : {dict(stats['category_counts'])}")
        print("=" * 60)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=0,
                        help="Nombre de queries (0 = toutes)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading queries...")
    queries = load_queries()
    if args.n_queries > 0:
        queries = queries[:args.n_queries]
    print(f"Loaded {len(queries)} queries.")

    print("Initializing ChromaStore...")
    store = ChromaStore()
    print("Initializing Llama...")
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb_rag = CPBNaiveRAG(naive_rag=naive_rag, architecture_name=RAG_NAME)

    # Accumulateurs pour le run summary
    stats: dict = {
        "n_detected": 0,
        "n_tp": 0,
        "n_fp": 0,
        "n_fn": 0,
        "n_sensitive": 0,
        "decision_counts": Counter(),
        "category_counts": Counter(),
    }

    for i, query_item in enumerate(queries, start=1):
        qid = query_item.get("query_id", f"q_{i}")
        sensitivity = query_item.get("sensitivity") or ""
        print(f"[{i}/{len(queries)}] {qid}  sensitivity={sensitivity or 'none'}")

        result = cpb_rag.run(query_item["query"], top_k=TOP_K)
        log_query_run(query_item, result)

        # Mettre à jour stats
        sad_detected = result.get("cpb_sad_detected", False)
        sad_decision = result.get("cpb_sad_decision", "pass")
        sad_categories = result.get("cpb_sad_categories", [])

        if sad_detected:
            stats["n_detected"] += 1
        if sensitivity in SENSITIVE_LABELS:
            stats["n_sensitive"] += 1

        stats["n_tp"] += _is_true_positive(sad_detected, sensitivity)
        stats["n_fp"] += _is_false_positive(sad_detected, sensitivity)
        stats["n_fn"] += _is_false_negative(sad_detected, sensitivity)
        stats["decision_counts"][sad_decision] += 1
        for cat in sad_categories:
            stats["category_counts"][cat] += 1

    log_summary_run(stats, n_queries=len(queries))
    print(f"\nDone. MLflow experiment: '{EXPERIMENT_NAME}'")
    print(f"UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
