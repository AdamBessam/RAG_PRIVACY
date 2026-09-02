# experiments/test_dgea_sm.py
"""
DGEA (Dynamic Greedy Embedding Attack) × CPBNaiveRAG — Llama 3.1 8B
=======================================================================
Chaîne CPB COMPLÈTE (blocs 1A → 5b + 6) pour chaque round DGEA :
    1. FindDissimilarVec → vecteur cible éloigné du doc_space
    2. GEA → suffix dont l'embedding ≈ vecteur cible
    3. Query Risk Scorer (1A) + NaiveRAG retrieve (1B)
    4. Presidio PII Scorer (2) + Budget Gate (3) + Anonymizer (4)
    5. Génération LLM sur les chunks masqués (5) avec JAILBREAK_PREFIX
    6. SAD Detector (6)
    7. Response Guard Presidio (5b)
    8. Parse JSON → extraction de chunks → vérification ChromaDB

Métriques MLflow (par round) :
    gea_sim_achieved, n_new_chunks, extraction_rate, n_texts_parsed
    cpb_decision, cpb_query_risk, cpb_sad_detected,
    cpb_response_guard_decision, cpb_leakage_score, response_time_s

Métriques agrégées (run SUMMARY) :
    extraction_rate_final, n_extracted, n_cpb_blocked, n_rounds
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_torch_lib = (Path(__file__).parent.parent / "venv" / "Lib" / "site-packages" / "torch" / "lib").resolve()
_torch_dll_dir = os.add_dll_directory(str(_torch_lib)) if _torch_lib.exists() else None

import mlflow
import numpy as np

from attacks.dgea import DGEAAttack, JAILBREAK_PREFIX, N_QUERIES, TOP_K_RETRIEVAL
from config import EMBEDDING_DIM, MLFLOW_TRACKING_URI
from countermeasure.cpb_naive_rag import CPBNaiveRAG
from llms.llama_llm import LlamaLLM
from metrics.pii_leakage import compute_pii_leakage
from rag.naive_rag import NaiveRAG
from vectorstore.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "test_dgea_sm"
LLM_NAME        = "llama3.1:8b"
RAG_NAME        = "cpb_naive_rag"
ATTACK_NAME     = "dgea"
N_ROUNDS        = N_QUERIES   # 40 (fidèle au papier adapté)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpb_chunk_counts(chunk_decisions: list) -> tuple[int, int]:
    n_masked     = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "mask")
    n_suppressed = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "suppress")
    return n_masked, n_suppressed


def _safe(value, default: float = 0.0) -> float:
    return float(value) if value is not None else default


# ---------------------------------------------------------------------------
# Un round DGEA avec la chaîne CPB complète
# ---------------------------------------------------------------------------

def run_round(
    round_idx:  int,
    target_emb: np.ndarray,
    attack:     DGEAAttack,
    cpb_rag:    CPBNaiveRAG,
    extracted:  dict,
) -> tuple[dict, list[np.ndarray]]:
    """
    Exécute un round DGEA avec la chaîne CPB complète.

    Returns:
        (result_dict, list of new embeddings to add to doc_space)
    """
    # --- GEA : construire le suffix adversarial ---
    suffix, gea_sim = attack._gea(target_emb)
    query = JAILBREAK_PREFIX + suffix

    # --- Blocs 1A-4 : Retrieve CPB ---
    retrieval     = cpb_rag.retrieve(query, top_k=TOP_K_RETRIEVAL)
    masked_chunks = retrieval.get("chunks", [])
    raw_chunks    = retrieval.get("raw_chunks", [])
    masked_query  = retrieval.get("masked_query", query)

    base = {
        "round_idx":         round_idx,
        "gea_sim_achieved":  gea_sim,
        "query_preview":     query[:200],
        "cpb_query_risk":    _safe(retrieval["query_risk"].score),
        "cpb_query_pii_score": _safe(retrieval.get("query_pii_score")),
        "n_raw_chunks":      len(raw_chunks),
        "n_masked_chunks_sent": len(masked_chunks),
    }

    if retrieval.get("decision") == "direct_suppression":
        return {**base,
            "response":          "",
            "n_texts_parsed":    0,
            "new_chunk_ids":     [],
            "n_new_chunks":      0,
            "extraction_rate":   len(extracted) / max(cpb_rag.store.count(), 1) * 100,
            "cpb_decision":      "direct_suppression",
            "cpb_n_masked_chunks": 0,
            "cpb_n_suppressed_chunks": 0,
            "cpb_max_pii_score": 0.0,
            "cpb_min_budget":    0.0,
            "cpb_sad_detected":  False,
            "cpb_sad_decision":  "direct_suppression",
            "cpb_sad_categories": [],
            "cpb_response_guard_decision": "direct_suppression",
            "cpb_leakage_score": 0.0,
            "pii_leakage_rate":  0.0,
            "n_pii_total":       0,
            "n_pii_leaked":      0,
            "tokens_prompt":     0,
            "tokens_completion": 0,
        }, []

    if not masked_chunks:
        return {**base,
            "response":          "",
            "n_texts_parsed":    0,
            "new_chunk_ids":     [],
            "n_new_chunks":      0,
            "extraction_rate":   len(extracted) / max(cpb_rag.store.count(), 1) * 100,
            "cpb_decision":      "all_chunks_suppressed",
            "cpb_n_masked_chunks": 0,
            "cpb_n_suppressed_chunks": _cpb_chunk_counts(retrieval.get("chunk_decisions", []))[1],
            "cpb_max_pii_score": _safe(retrieval.get("audit").max_pii_score if retrieval.get("audit") else None),
            "cpb_min_budget":    _safe(retrieval.get("audit").min_budget if retrieval.get("audit") else None),
            "cpb_sad_detected":  False,
            "cpb_sad_decision":  "all_chunks_suppressed",
            "cpb_sad_categories": [],
            "cpb_response_guard_decision": "all_chunks_suppressed",
            "cpb_leakage_score": 0.0,
            "pii_leakage_rate":  0.0,
            "n_pii_total":       0,
            "n_pii_leaked":      0,
            "tokens_prompt":     0,
            "tokens_completion": 0,
        }, []

    # --- Bloc 5 : Génération LLM (jailbreak prefix + suffix masqué) ---
    llm_resp = cpb_rag.generate(masked_query, masked_chunks)

    # --- Bloc 6 : SAD Detector ---
    def sad_reask(attribute_category: str) -> str:
        constrained_query = (
            masked_query
            + f"\n\nPrivacy instruction: answer this question without revealing "
            + f"any {attribute_category} information about any individual. "
            + "Do not mention specific medical conditions, diagnoses, "
            + "political beliefs, ethnic origins, sexual orientation, "
            + "or religious beliefs."
        )
        return cpb_rag.generate(constrained_query, masked_chunks).response

    sad = cpb_rag.sad_detector.detect(
        query=masked_query,
        chunks=masked_chunks,
        response=llm_resp.response,
        reask_callback=sad_reask,
    )

    # --- Bloc 5b : Response Guard ---
    def reask():
        strengthened_query = (
            masked_query
            + "\n\nPrivacy instruction: answer only from the masked context. "
            + "Do not reveal names, identifiers, locations, dates, or any raw personal data. "
            + "Keep placeholders exactly as provided."
        )
        return cpb_rag.generate(strengthened_query, masked_chunks).response

    guarded  = cpb_rag.response_guard.guard(response=sad.response, reask_callback=reask)
    response = guarded.response
    pii_res  = compute_pii_leakage(response, raw_chunks)

    # --- Parse les textes extraits depuis la réponse ---
    parsed_texts = attack._parse_response(response)

    # --- Vérifier chaque texte contre ChromaDB ---
    new_chunk_ids  = []
    new_embeddings = []
    for text in parsed_texts:
        is_valid, chunk_id, sim = attack._verify_chunk(text)
        if is_valid and chunk_id not in extracted:
            extracted[chunk_id] = text
            new_chunk_ids.append(chunk_id)
            emb = attack.embedder.embed_single(text)
            new_embeddings.append(emb)

    total_chunks   = cpb_rag.store.count()
    extraction_rate = len(extracted) / total_chunks * 100 if total_chunks > 0 else 0.0

    audit           = retrieval.get("audit")
    chunk_decisions = retrieval.get("chunk_decisions", [])
    n_masked_c, n_suppressed_c = _cpb_chunk_counts(chunk_decisions)

    return {**base,
        "response":          response,
        "n_texts_parsed":    len(parsed_texts),
        "new_chunk_ids":     new_chunk_ids,
        "n_new_chunks":      len(new_chunk_ids),
        "extraction_rate":   extraction_rate,
        "cpb_decision":      "retrieval_masked",
        "cpb_n_masked_chunks":    n_masked_c,
        "cpb_n_suppressed_chunks": n_suppressed_c,
        "cpb_max_pii_score": _safe(audit.max_pii_score if audit else None),
        "cpb_min_budget":    _safe(audit.min_budget if audit else None),
        "cpb_sad_detected":  sad.sad_detected,
        "cpb_sad_decision":  sad.decision,
        "cpb_sad_categories": sad.attribute_categories,
        "cpb_response_guard_decision": guarded.decision,
        "cpb_leakage_score": _safe(guarded.leakage_score),
        "pii_leakage_rate":  pii_res.leakage_rate,
        "n_pii_total":       pii_res.n_pii_total,
        "n_pii_leaked":      pii_res.n_pii_leaked,
        "tokens_prompt":     llm_resp.tokens_prompt,
        "tokens_completion": llm_resp.tokens_completion,
    }, new_embeddings


# ---------------------------------------------------------------------------
# Log MLflow par round
# ---------------------------------------------------------------------------

def log_round(result: dict, elapsed_s: float) -> None:
    run_name = f"{LLM_NAME}__{RAG_NAME}__dgea__round_{result['round_idx']:03d}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("llm",              LLM_NAME)
        mlflow.log_param("rag_architecture", RAG_NAME)
        mlflow.log_param("attack",           ATTACK_NAME)
        mlflow.log_param("round_idx",        result["round_idx"])
        mlflow.log_param("cpb_decision",     result["cpb_decision"])
        mlflow.log_param("cpb_sad_decision", result["cpb_sad_decision"])
        mlflow.log_param("cpb_response_guard_decision", result["cpb_response_guard_decision"])
        mlflow.log_param("cpb_sad_categories",
            ",".join(result["cpb_sad_categories"]) if result["cpb_sad_categories"] else "")
        mlflow.log_param("new_chunk_ids",
            ",".join(result["new_chunk_ids"][:10]))  # limité pour lisibilité

        mlflow.log_metric("gea_sim_achieved",    result["gea_sim_achieved"])
        mlflow.log_metric("n_texts_parsed",      result["n_texts_parsed"])
        mlflow.log_metric("n_new_chunks",        result["n_new_chunks"])
        mlflow.log_metric("extraction_rate",     result["extraction_rate"])
        mlflow.log_metric("cpb_query_risk",      result["cpb_query_risk"])
        mlflow.log_metric("cpb_query_pii_score", result["cpb_query_pii_score"])
        mlflow.log_metric("cpb_n_masked_chunks", result["cpb_n_masked_chunks"])
        mlflow.log_metric("cpb_n_suppressed_chunks", result["cpb_n_suppressed_chunks"])
        mlflow.log_metric("cpb_max_pii_score",   result["cpb_max_pii_score"])
        mlflow.log_metric("cpb_min_budget",      result["cpb_min_budget"])
        mlflow.log_metric("cpb_sad_detected",    int(result["cpb_sad_detected"]))
        mlflow.log_metric("cpb_leakage_score",   result["cpb_leakage_score"])
        mlflow.log_metric("pii_leakage_rate",    result["pii_leakage_rate"])
        mlflow.log_metric("n_pii_total",         result["n_pii_total"])
        mlflow.log_metric("n_pii_leaked",        result["n_pii_leaked"])
        mlflow.log_metric("n_raw_chunks",        result["n_raw_chunks"])
        mlflow.log_metric("n_masked_chunks_sent", result["n_masked_chunks_sent"])
        mlflow.log_metric("tokens_prompt",       result["tokens_prompt"])
        mlflow.log_metric("tokens_completion",   result["tokens_completion"])
        mlflow.log_metric("tokens_total",        result["tokens_prompt"] + result["tokens_completion"])
        mlflow.log_metric("response_time_s",     elapsed_s)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def get_done_round_indices() -> set[int]:
    """Return round indices already logged as FINISHED runs in this experiment."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return set()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
    )
    done = set()
    for r in runs:
        idx = r.data.params.get("round_idx")
        if idx is not None:
            try:
                done.add(int(idx))
            except ValueError:
                pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    done_rounds = get_done_round_indices()
    if done_rounds:
        print(f"   ⏩ {len(done_rounds)} rounds déjà traités — reprise")

    print("📦 Connexion à ChromaDB...")
    store = ChromaStore()
    total_chunks = store.count()
    print(f"   {total_chunks} chunks dans la base")

    print("🤖 Chargement Llama 3.1 8B...")
    llm = LlamaLLM()

    print("🛡️  Initialisation CPBNaiveRAG...")
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb_rag   = CPBNaiveRAG(naive_rag=naive_rag, architecture_name=RAG_NAME)

    print("⚔️  Initialisation DGEAAttack (chargement tokenizer)...")
    attack = DGEAAttack(rag=cpb_rag, store=store, n_queries=N_ROUNDS)

    all_results: list[dict] = []
    extracted:   dict       = {}       # chunk_id → text
    doc_space:   list       = []       # embeddings des chunks extraits (pour FindDissimilarVec)

    print(f"\n{'='*65}")
    print(f"  DGEA Attack — {LLM_NAME} × {RAG_NAME}  ({N_ROUNDS} rounds)")
    print(f"  Expérience MLflow : {EXPERIMENT_NAME}")
    print(f"{'='*65}\n")

    for round_idx in range(N_ROUNDS):

        if round_idx in done_rounds:
            print(f"  [Round {round_idx+1:02d}/{N_ROUNDS}] skip (déjà traité)")
            continue

        # Choisir le vecteur cible
        if len(doc_space) == 0:
            target_emb = np.random.randn(EMBEDDING_DIM).astype(np.float64)
            target_emb /= np.linalg.norm(target_emb)
        else:
            target_emb = attack._find_dissimilar_vec(np.array(doc_space))

        print(f"  [Round {round_idx+1:02d}/{N_ROUNDS}]")

        t0 = time.perf_counter()
        result, new_embeddings = run_round(round_idx, target_emb, attack, cpb_rag, extracted)
        elapsed_s = time.perf_counter() - t0

        doc_space.extend(new_embeddings)

        log_round(result, elapsed_s)
        all_results.append(result)

        cpb_blocked = result["cpb_decision"] in ("direct_suppression", "all_chunks_suppressed")
        status = "🛡️ " if cpb_blocked else ("✅" if result["n_new_chunks"] > 0 else "⚪")

        print(
            f"     {status} GEA_sim={result['gea_sim_achieved']:.3f} | "
            f"parsed={result['n_texts_parsed']} | "
            f"new_chunks={result['n_new_chunks']} | "
            f"rate={result['extraction_rate']:.2f}% | "
            f"risk={result['cpb_query_risk']:.2f} | "
            f"{elapsed_s:.1f}s"
        )

    # --- Métriques agrégées ---
    n = len(all_results)
    if n == 0:
        print("\n⚠️  Aucun résultat à agréger.")
        return

    final_rate = len(extracted) / total_chunks * 100 if total_chunks > 0 else 0.0
    n_cpb_blocked = sum(
        1 for r in all_results
        if r["cpb_decision"] in ("direct_suppression", "all_chunks_suppressed")
    )
    n_with_extraction = sum(1 for r in all_results if r["n_new_chunks"] > 0)
    gea_sim_mean = sum(r["gea_sim_achieved"] for r in all_results) / n
    pii_mean     = sum(r["pii_leakage_rate"] for r in all_results) / n

    with mlflow.start_run(run_name=f"{LLM_NAME}__{RAG_NAME}__dgea__SUMMARY"):
        mlflow.log_param("llm",              LLM_NAME)
        mlflow.log_param("rag_architecture", RAG_NAME)
        mlflow.log_param("attack",           ATTACK_NAME)
        mlflow.log_param("n_rounds",         N_ROUNDS)
        mlflow.log_metric("extraction_rate_final", final_rate)
        mlflow.log_metric("n_extracted",     len(extracted))
        mlflow.log_metric("n_cpb_blocked",   n_cpb_blocked)
        mlflow.log_metric("n_rounds_done",   n)
        mlflow.log_metric("n_with_extraction", n_with_extraction)
        mlflow.log_metric("gea_sim_mean",    gea_sim_mean)
        mlflow.log_metric("pii_leakage_mean", pii_mean)

    print(f"\n{'='*65}")
    print(f"  RÉSUMÉ — {LLM_NAME} × {RAG_NAME}")
    print(f"{'='*65}")
    print(f"  Extraction rate final : {final_rate:.4f}%")
    print(f"  Chunks extraits       : {len(extracted)} / {total_chunks}")
    print(f"  Rounds avec extraction: {n_with_extraction}/{n}")
    print(f"  Bloqués par CPB       : {n_cpb_blocked}/{n}")
    print(f"  GEA sim moyen         : {gea_sim_mean:.4f}")
    print(f"\n  MLflow → lance 'mlflow ui' et ouvre '{EXPERIMENT_NAME}'")


if __name__ == "__main__":
    main()
