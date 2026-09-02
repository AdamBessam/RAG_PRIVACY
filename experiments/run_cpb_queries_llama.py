# experiments/run_cpb_queries_llama.py
"""
Evaluate CPBNaiveRAG on the existing benchmark queries with Llama.

Important leakage rule:
    PII leakage is computed by comparing the final CPB response against the
    raw chunks retrieved before masking, not against the masked chunks sent to
    the LLM.

This keeps the metric honest: if the final answer contains a raw PII from the
retrieved context, it is counted as leakage even though the LLM received masked
chunks.
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from config import MLFLOW_TRACKING_URI, QUERY_LOG_MAX_CHARS, TOP_K
from countermeasure.cpb_naive_rag import CPBNaiveRAG
from data.query_generator import load_queries
from llms.llama_llm import LlamaLLM
from metrics.pii_leakage import compute_pii_leakage
from rag.naive_rag import NaiveRAG
from vectorstore.chroma_store import ChromaStore


CPB_EXPERIMENT_NAME = os.getenv("CPB_EXPERIMENT_NAME", "sad_test_v5")
CPB_MAX_QUERIES = int(os.getenv("CPB_MAX_QUERIES", "0") or "0")
CPB_START_INDEX = int(os.getenv("CPB_START_INDEX", "0") or "0")
ATTACK_NAME = "cpb_queries_eval"
LLM_NAME = "llama3.1:8b"
RAG_NAME = "cpb_naive_rag"


def _target_entity_leaked(response: str, target_entity: str | None) -> int:
    if not target_entity:
        return 0
    return int(target_entity.lower() in response.lower())


def _cpb_chunk_counts(chunk_decisions: list) -> tuple[int, int]:
    n_masked = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "mask")
    n_suppressed = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "suppress")
    return n_masked, n_suppressed


def _safe_metric(value, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def log_query_run(query_item: dict, result: dict, elapsed_s: float) -> None:
    raw_chunks = result.get("raw_chunks", [])
    masked_chunks = result.get("chunks", [])

    # Core rule: leakage is measured against pre-masking raw chunks.
    pii = compute_pii_leakage(
        response=result["response"],
        chunks=raw_chunks,
    )

    audit = result.get("cpb_audit")
    chunk_decisions = result.get("cpb_chunk_decisions", [])
    n_masked, n_suppressed = _cpb_chunk_counts(chunk_decisions)

    run_name = f"{LLM_NAME}__{RAG_NAME}__{query_item.get('query_id', 'unknown')}"
    with mlflow.start_run(run_name=run_name):
        # Params
        mlflow.log_param("llm", LLM_NAME)
        mlflow.log_param("rag_architecture", RAG_NAME)
        mlflow.log_param("attack", ATTACK_NAME)
        mlflow.log_param("query_id", query_item.get("query_id", "unknown"))
        mlflow.log_param("query_type", query_item.get("query_type", "unknown"))
        mlflow.log_param("question_type", query_item.get("query_type", "unknown"))
        mlflow.log_param("query", (result.get("cpb_masked_query") or query_item["query"])[:QUERY_LOG_MAX_CHARS])
        mlflow.log_param("query_logged_is_masked", True)
        mlflow.log_param("has_target_entity", bool(query_item.get("target_entity")))
        mlflow.log_param("entity_type", query_item.get("entity_type") or "")
        mlflow.log_param("sensitivity", query_item.get("sensitivity") or "")
        mlflow.log_param("doc_id", query_item.get("doc_id") or "")
        mlflow.log_param("response_preview", result["response"][:QUERY_LOG_MAX_CHARS])
        mlflow.log_param("cpb_response_guard_decision", result.get("cpb_response_guard_decision", ""))

        # Standard metrics
        mlflow.log_metric("tokens_prompt", _safe_metric(result.get("tokens_prompt")))
        mlflow.log_metric("tokens_completion", _safe_metric(result.get("tokens_completion")))
        mlflow.log_metric("tokens_total", _safe_metric(result.get("tokens_total")))
        mlflow.log_metric("cost_usd", _safe_metric(result.get("cost_usd")))

        # PII leakage computed against raw chunks.
        mlflow.log_metric("pii_leakage_rate", pii.leakage_rate)
        mlflow.log_metric("n_pii_total", pii.n_pii_total)
        mlflow.log_metric("n_pii_leaked", pii.n_pii_leaked)
        mlflow.log_metric(
            "target_entity_leaked",
            _target_entity_leaked(result["response"], query_item.get("target_entity")),
        )

        # CPB metrics
        mlflow.log_metric("n_raw_chunks", len(raw_chunks))
        mlflow.log_metric("n_masked_chunks", len(masked_chunks))
        mlflow.log_metric("cpb_query_risk", _safe_metric(result.get("cpb_query_risk")))
        mlflow.log_metric("cpb_query_pii_score", _safe_metric(result.get("cpb_query_pii_score")))
        mlflow.log_metric("cpb_query_pii_findings_count", _safe_metric(result.get("cpb_query_pii_findings_count")))
        mlflow.log_metric("cpb_query_pii_replacements", _safe_metric(result.get("cpb_query_pii_replacements")))
        mlflow.log_metric("cpb_n_masked_chunks", n_masked)
        mlflow.log_metric("cpb_n_suppressed_chunks", n_suppressed)

        if audit is not None:
            mlflow.log_metric("cpb_max_pii_score", audit.max_pii_score)
            mlflow.log_metric("cpb_min_budget", audit.min_budget)
            mlflow.log_metric("cpb_response_leakage_score", audit.leakage_score)

        # SAD metrics
        mlflow.log_metric("cpb_sad_detected", int(result.get("cpb_sad_detected", False)))
        mlflow.log_metric("cpb_sad_confidence", _safe_metric(result.get("cpb_sad_confidence")))
        mlflow.log_metric("cpb_sad_filter", _safe_metric(result.get("cpb_sad_filter")))
        mlflow.log_param("cpb_sad_decision", result.get("cpb_sad_decision", "pass"))
        mlflow.log_param("cpb_sad_categories", ",".join(c for c in (result.get("cpb_sad_categories") or []) if c is not None))

        # S5 semantic signal
        signals = result.get("cpb_query_risk_signals", {})
        mlflow.log_metric("cpb_s5_semantic", _safe_metric(signals.get("s5_semantic")))

        # Response time
        mlflow.log_metric("response_time_s", elapsed_s)


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(CPB_EXPERIMENT_NAME)

    print("Loading queries...")
    queries = load_queries()
    if CPB_MAX_QUERIES > 0:
        queries = queries[:CPB_MAX_QUERIES]
    if CPB_START_INDEX > 0:
        queries = queries[CPB_START_INDEX:]
    print(f"Loaded {len(queries)} queries (starting from index {CPB_START_INDEX}).")

    print("Initializing ChromaStore...")
    store = ChromaStore()

    print("Initializing Llama...")
    llm = LlamaLLM()

    print("Initializing CPBNaiveRAG...")
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb_rag = CPBNaiveRAG(naive_rag=naive_rag, architecture_name=RAG_NAME)

    for i, query_item in enumerate(queries, start=1):
        print(
            f"[{i}/{len(queries)}] "
            f"{query_item.get('query_id')} ({query_item.get('query_type')})"
        )
        t0 = time.perf_counter()
        result = cpb_rag.run(query_item["query"], top_k=TOP_K)
        elapsed_s = time.perf_counter() - t0
        print(f"    → {elapsed_s:.1f}s")
        log_query_run(query_item, result, elapsed_s)

    print(f"Done. Runs logged in MLflow experiment: {CPB_EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
