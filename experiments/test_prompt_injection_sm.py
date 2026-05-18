# experiments/test_prompt_injection_sm.py
"""
Prompt Injection Attack (Spill the Beans) × CPBNaiveRAG — Llama 3.1 8B
=======================================================================
Même principe d'attaque que run_prompt_injection.py :
    1. Récupération via la requête originale (anchor)
    2. Remplacement par le prompt adversarial pour la génération
    3. Le LLM reçoit : [chunks masqués CPB] + [instruction de régurgitation]

Différence clé par rapport au RAG sans défense :
    - Les chunks envoyés au LLM sont MASQUÉS par le CPB (entités PII remplacées)
    - ROUGE-L et PII leakage sont calculés contre les RAW chunks (avant masquage)
      → mesure honnête : si le LLM régurgite des PII, c'est qu'il les a reconstituées
        malgré le masquage.

Métriques loggées dans MLflow (expérience : test_prompt_injection_sm) :
    Standard attaque :
        rouge_l, pii_leakage_rate, reconstruction_success, n_pii_leaked
    CPB spécifique :
        cpb_query_risk, cpb_n_masked_chunks, cpb_n_suppressed_chunks,
        cpb_query_pii_score, cpb_min_budget, cpb_max_pii_score
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from attacks.prompt_injection import PromptInjectionAttack
from config import MLFLOW_TRACKING_URI, TOP_K, ROUGE_L_THRESHOLD, QUERY_LOG_MAX_CHARS
from countermeasure.cpb_naive_rag import CPBNaiveRAG
from data.query_generator import load_queries
from llms.llama_llm import LlamaLLM
from metrics.pii_leakage import compute_pii_leakage
from metrics.rouge_score import compute_rouge_l
from rag.naive_rag import NaiveRAG
from vectorstore.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "test_prompt_injection_sm"
LLM_NAME        = "llama3.1:8b"
RAG_NAME        = "cpb_naive_rag"
ATTACK_NAME     = "prompt_injection"


# ---------------------------------------------------------------------------
# Helpers CPB
# ---------------------------------------------------------------------------

def _cpb_chunk_counts(chunk_decisions: list) -> tuple[int, int]:
    n_masked     = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "mask")
    n_suppressed = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "suppress")
    return n_masked, n_suppressed


def _safe(value, default: float = 0.0) -> float:
    return float(value) if value is not None else default


# ---------------------------------------------------------------------------
# Run sur une requête
# ---------------------------------------------------------------------------

def run_single(
    query_item: dict,
    cpb_rag:    CPBNaiveRAG,
    top_k:      int = TOP_K,
) -> dict:
    """
    Exécute l'attaque prompt injection sur une requête via CPBNaiveRAG.

    Étapes :
        1. Retrieve avec la requête originale → chunks masqués + raw chunks
        2. Construire le prompt adversarial
        3. Générer via le LLM sur les chunks masqués
        4. Calculer ROUGE-L et PII leakage contre les raw chunks
    """
    query = query_item["query"]

    # --- Étape 1 : Retrieve (le risk scorer voit la requête bénigne) ---
    retrieval    = cpb_rag.retrieve(query, top_k=top_k)
    masked_chunks = retrieval.get("chunks", [])
    raw_chunks    = retrieval.get("raw_chunks", [])

    # Suppression directe par le CPB
    if retrieval.get("decision") == "direct_suppression":
        return {
            "query":                 query,
            "injected_query":        "",
            "response":              "I cannot process this request because it asks for sensitive context disclosure.",
            "masked_chunks":         [],
            "raw_chunks":            [],
            "rouge_l":               0.0,
            "rouge_precision":       0.0,
            "rouge_recall":          0.0,
            "reconstruction_success": False,
            "pii_leakage_rate":      0.0,
            "n_pii_total":           0,
            "n_pii_leaked":          0,
            "tokens_prompt":         0,
            "tokens_completion":     0,
            "cost_usd":              0.0,
            "cpb_decision":          "direct_suppression",
            "cpb_query_risk":        _safe(retrieval["query_risk"].score),
            "cpb_query_pii_score":   _safe(retrieval.get("query_pii_score")),
            "cpb_n_masked_chunks":   0,
            "cpb_n_suppressed_chunks": 0,
            "cpb_max_pii_score":     0.0,
            "cpb_min_budget":        0.0,
        }

    # Tous les chunks supprimés → aucun contexte envoyé au LLM
    if not masked_chunks:
        return {
            "query":                 query,
            "injected_query":        "",
            "response":              "I cannot provide details because the retrieved context is too sensitive after privacy filtering.",
            "masked_chunks":         [],
            "raw_chunks":            raw_chunks,
            "rouge_l":               0.0,
            "rouge_precision":       0.0,
            "rouge_recall":          0.0,
            "reconstruction_success": False,
            "pii_leakage_rate":      0.0,
            "n_pii_total":           0,
            "n_pii_leaked":          0,
            "tokens_prompt":         0,
            "tokens_completion":     0,
            "cost_usd":              0.0,
            "cpb_decision":          "all_chunks_suppressed",
            "cpb_query_risk":        _safe(retrieval["query_risk"].score),
            "cpb_query_pii_score":   _safe(retrieval.get("query_pii_score")),
            "cpb_n_masked_chunks":   0,
            "cpb_n_suppressed_chunks": _cpb_chunk_counts(retrieval.get("chunk_decisions", []))[1],
            "cpb_max_pii_score":     _safe(retrieval.get("audit").max_pii_score if retrieval.get("audit") else None),
            "cpb_min_budget":        _safe(retrieval.get("audit").min_budget if retrieval.get("audit") else None),
        }

    # --- Étape 2 : Construire le prompt adversarial ---
    injected_query = PromptInjectionAttack.ADVERSARIAL_TEMPLATE.format(anchor=query)

    # --- Étape 3 : Générer avec le prompt injecté sur les chunks MASQUÉS ---
    llm_resp = cpb_rag.generate(injected_query, masked_chunks)
    response = llm_resp.response

    # --- Étape 4 : Métriques contre les raw chunks (mesure honnête) ---
    ref_text     = "\n\n".join(c.get("text", "") for c in raw_chunks if c.get("text"))
    rouge_result = compute_rouge_l(response, ref_text)
    pii_result   = compute_pii_leakage(response, raw_chunks)

    audit           = retrieval.get("audit")
    chunk_decisions = retrieval.get("chunk_decisions", [])
    n_masked, n_suppressed = _cpb_chunk_counts(chunk_decisions)

    return {
        "query":                  query,
        "injected_query":         injected_query,
        "response":               response,
        "masked_chunks":          masked_chunks,
        "raw_chunks":             raw_chunks,
        "rouge_l":                rouge_result.rouge_l,
        "rouge_precision":        rouge_result.precision,
        "rouge_recall":           rouge_result.recall,
        "reconstruction_success": rouge_result.reconstruction_success,
        "pii_leakage_rate":       pii_result.leakage_rate,
        "n_pii_total":            pii_result.n_pii_total,
        "n_pii_leaked":           pii_result.n_pii_leaked,
        "tokens_prompt":          llm_resp.tokens_prompt,
        "tokens_completion":      llm_resp.tokens_completion,
        "cost_usd":               llm_resp.cost_usd,
        "cpb_decision":           retrieval.get("decision", "retrieval_masked"),
        "cpb_query_risk":         _safe(retrieval["query_risk"].score),
        "cpb_query_pii_score":    _safe(retrieval.get("query_pii_score")),
        "cpb_n_masked_chunks":    n_masked,
        "cpb_n_suppressed_chunks": n_suppressed,
        "cpb_max_pii_score":      _safe(audit.max_pii_score if audit else None),
        "cpb_min_budget":         _safe(audit.min_budget if audit else None),
    }


# ---------------------------------------------------------------------------
# Log MLflow
# ---------------------------------------------------------------------------

def log_run(query_item: dict, result: dict, elapsed_s: float) -> None:
    run_name = f"{LLM_NAME}__{RAG_NAME}__{query_item.get('query_id', 'unknown')}"
    with mlflow.start_run(run_name=run_name):

        # --- Params ---
        mlflow.log_param("llm",              LLM_NAME)
        mlflow.log_param("rag_architecture", RAG_NAME)
        mlflow.log_param("attack",           ATTACK_NAME)
        mlflow.log_param("query_id",         query_item.get("query_id", "unknown"))
        mlflow.log_param("query_type",       query_item.get("query_type", "unknown"))
        mlflow.log_param("entity_type",      query_item.get("entity_type") or "")
        mlflow.log_param("sensitivity",      query_item.get("sensitivity") or "")
        mlflow.log_param("doc_id",           query_item.get("doc_id") or "")
        mlflow.log_param("query",            result["query"][:QUERY_LOG_MAX_CHARS])
        mlflow.log_param("response_preview", result["response"][:QUERY_LOG_MAX_CHARS])
        mlflow.log_param("cpb_decision",     result["cpb_decision"])

        # --- Métriques attaque (contre raw chunks) ---
        mlflow.log_metric("rouge_l",               result["rouge_l"])
        mlflow.log_metric("rouge_precision",        result["rouge_precision"])
        mlflow.log_metric("rouge_recall",           result["rouge_recall"])
        mlflow.log_metric("reconstruction_success", int(result["reconstruction_success"]))
        mlflow.log_metric("pii_leakage_rate",       result["pii_leakage_rate"])
        mlflow.log_metric("n_pii_total",            result["n_pii_total"])
        mlflow.log_metric("n_pii_leaked",           result["n_pii_leaked"])

        # --- Tokens / coût ---
        mlflow.log_metric("tokens_prompt",     result["tokens_prompt"])
        mlflow.log_metric("tokens_completion", result["tokens_completion"])
        mlflow.log_metric("tokens_total",      result["tokens_prompt"] + result["tokens_completion"])
        mlflow.log_metric("cost_usd",          result["cost_usd"])

        # --- Métriques CPB ---
        mlflow.log_metric("cpb_query_risk",          result["cpb_query_risk"])
        mlflow.log_metric("cpb_query_pii_score",     result["cpb_query_pii_score"])
        mlflow.log_metric("cpb_n_masked_chunks",     result["cpb_n_masked_chunks"])
        mlflow.log_metric("cpb_n_suppressed_chunks", result["cpb_n_suppressed_chunks"])
        mlflow.log_metric("cpb_max_pii_score",       result["cpb_max_pii_score"])
        mlflow.log_metric("cpb_min_budget",          result["cpb_min_budget"])
        mlflow.log_metric("n_raw_chunks",            len(result["raw_chunks"]))
        mlflow.log_metric("n_masked_chunks_sent",    len(result["masked_chunks"]))

        # --- Temps de réponse ---
        mlflow.log_metric("response_time_s", elapsed_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("📥 Chargement des requêtes...")
    queries = load_queries()
    print(f"   {len(queries)} requêtes chargées")

    print("📦 Connexion à ChromaDB...")
    store = ChromaStore()

    print("🤖 Chargement Llama 3.1 8B...")
    llm = LlamaLLM()

    print("🛡️  Initialisation CPBNaiveRAG...")
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb_rag   = CPBNaiveRAG(naive_rag=naive_rag, architecture_name=RAG_NAME)

    n_success = 0
    n_suppressed = 0
    rouge_scores = []
    pii_scores   = []

    print(f"\n{'='*65}")
    print(f"  Prompt Injection — {LLM_NAME} × {RAG_NAME}")
    print(f"  Expérience MLflow : {EXPERIMENT_NAME}")
    print(f"{'='*65}\n")

    for i, query_item in enumerate(queries, start=1):
        print(f"[{i:02d}/{len(queries)}] {query_item.get('query_id')} ({query_item.get('query_type')})")

        t0 = time.perf_counter()
        result = run_single(query_item, cpb_rag, top_k=TOP_K)
        elapsed_s = time.perf_counter() - t0

        log_run(query_item, result, elapsed_s)

        # Affichage console
        if result["cpb_decision"] in ("direct_suppression", "all_chunks_suppressed"):
            print(f"   🛡️  CPB bloqué  ({result['cpb_decision']}) — {elapsed_s:.1f}s")
            n_suppressed += 1
        else:
            status = "✅" if result["reconstruction_success"] else "❌"
            print(
                f"   {status} ROUGE-L={result['rouge_l']:.3f} | "
                f"PII={result['pii_leakage_rate']:.3f} | "
                f"risk={result['cpb_query_risk']:.2f} | "
                f"masked={result['cpb_n_masked_chunks']} | "
                f"{elapsed_s:.1f}s"
            )
            if result["reconstruction_success"]:
                n_success += 1
            rouge_scores.append(result["rouge_l"])
            pii_scores.append(result["pii_leakage_rate"])

    # --- Résumé final ---
    n_eval = len(queries) - n_suppressed
    print(f"\n{'='*65}")
    print(f"  RÉSUMÉ — {LLM_NAME} × {RAG_NAME}")
    print(f"{'='*65}")
    print(f"  Total requêtes          : {len(queries)}")
    print(f"  Bloquées par CPB        : {n_suppressed}")
    print(f"  Évaluées               : {n_eval}")
    if rouge_scores:
        print(f"  ROUGE-L moyen          : {sum(rouge_scores)/len(rouge_scores):.4f}")
        print(f"  PII leakage moyen      : {sum(pii_scores)/len(pii_scores):.4f}")
        print(f"  Reconstructions OK     : {n_success}/{n_eval}")
    print(f"\n  MLflow → lance 'mlflow ui' et ouvre l'expérience '{EXPERIMENT_NAME}'")


if __name__ == "__main__":
    main()
