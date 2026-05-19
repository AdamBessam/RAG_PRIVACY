# experiments/test_mia_sm.py
"""
MIA (Membership Inference Attack) × CPBNaiveRAG — Llama 3.1 8B
=======================================================================
Chaîne CPB COMPLÈTE (blocs 1A → 5b + 6) pour chaque sample MIA :
    1. Build attack prompt  : "Does this appear in the context? Yes/No."
    2. Query Risk Scorer (1A) + NaiveRAG retrieve (1B)
    3. Presidio PII Scorer (2) + Budget Gate (3) + Anonymizer (4)
    4. Génération LLM sur les chunks masqués (5)
    5. SAD Detector (6)
    6. Response Guard Presidio (5b)
    7. Parse Yes/No → membership prediction

Members  : N_PAIRS chunks réels de ChromaDB
Non-members : textes du split test/validation HuggingFace (non indexés)
              → fallback : deuxième moitié du split train (positions décalées)

Métriques MLflow (par sample) :
    is_member, predicted_member, correct, cpb_decision,
    cpb_query_risk, cpb_sad_detected, cpb_response_guard_decision,
    cpb_leakage_score, tokens_total, response_time_s

Métriques agrégées (run SUMMARY) :
    tpr, fpr, auc_roc, n_cpb_blocked, n_missing_yes_no
"""
import os
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_torch_lib = (Path(__file__).parent.parent / "venv" / "Lib" / "site-packages" / "torch" / "lib").resolve()
_torch_dll_dir = os.add_dll_directory(str(_torch_lib)) if _torch_lib.exists() else None

import mlflow
import numpy as np

from attacks.membership_inference import (
    MembershipInferenceAttack,
    build_attack_prompt,
    _parse_response,
    DEFAULT_PROMPT_ID,
)
from config import (
    DATASET_NAME, MEMBERSHIP_INFERENCE_N_PAIRS,
    MLFLOW_TRACKING_URI, TOP_K,
)
from countermeasure.cpb_naive_rag import CPBNaiveRAG
from llms.llama_llm import LlamaLLM
from metrics.pii_leakage import compute_pii_leakage
from rag.naive_rag import NaiveRAG
from vectorstore.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "test_mia_sm"
LLM_NAME        = "llama3.1:8b"
RAG_NAME        = "cpb_naive_rag"
ATTACK_NAME     = "membership_inference"
N_PAIRS         = MEMBERSHIP_INFERENCE_N_PAIRS   # 20 members + 20 non-members
PROMPT_ID       = DEFAULT_PROMPT_ID              # prompt variant (paper best: 2)


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
# Chargement des samples
# ---------------------------------------------------------------------------

def load_member_samples(store: ChromaStore, n: int) -> list[dict]:
    """Récupère n chunks réels depuis ChromaDB (ground truth = member)."""
    result = store.collection.get(limit=n * 3)   # sur-échantillon puis shuffle
    docs = result.get("documents", [])
    ids  = result.get("ids", [])
    pairs = list(zip(ids, docs))
    random.shuffle(pairs)
    return [{"chunk_id": cid, "text": text} for cid, text in pairs[:n]]


def load_non_member_texts(n: int) -> list[str]:
    """
    Charge n textes qui ne sont PAS dans ChromaDB.
    Stratégie 1 : split test/validation HuggingFace (jamais indexé).
    Stratégie 2 : fallback → deuxième moitié du split train, position décalée.
    """
    from datasets import load_dataset

    for split in ("test", "validation", "dev"):
        try:
            ds = load_dataset(DATASET_NAME, split=split)
            texts = []
            for sample in ds:
                t = sample.get("text", "")[:500].strip()
                if len(t) > 50:
                    texts.append(t)
                if len(texts) >= n:
                    break
            if texts:
                print(f"   Non-members : split '{split}' ({len(texts)} textes)")
                return texts[:n]
        except Exception:
            continue

    # Fallback : train split, position 500-1000 chars (hors fenêtre de chunking)
    print("   Non-members : fallback train (position décalée)")
    ds = load_dataset(DATASET_NAME, split="train")
    all_docs = list(ds)
    random.shuffle(all_docs)
    texts = []
    for sample in all_docs:
        full = sample.get("text", "")
        t = full[500:1000].strip()   # hors de la première fenêtre de chunk
        if len(t) > 50:
            texts.append(t)
        if len(texts) >= n:
            break
    print(f"   Non-members : {len(texts)} textes chargés (fallback)")
    return texts[:n]


# ---------------------------------------------------------------------------
# Un sample MIA avec la chaîne CPB complète
# ---------------------------------------------------------------------------

def run_sample(
    text:      str,
    is_member: bool,
    sample_id: str,
    cpb_rag:   CPBNaiveRAG,
) -> dict:
    """
    Exécute l'attaque MIA sur un sample avec la chaîne CPB complète.
    """
    attack_prompt = build_attack_prompt(text, PROMPT_ID)

    # --- Blocs 1A-4 : Retrieve + masquage ---
    retrieval     = cpb_rag.retrieve(attack_prompt, top_k=TOP_K)
    masked_chunks = retrieval.get("chunks", [])
    raw_chunks    = retrieval.get("raw_chunks", [])
    masked_query  = retrieval.get("masked_query", attack_prompt)

    base = {
        "sample_id":    sample_id,
        "text_preview": text[:100],
        "is_member":    is_member,
        "attack_prompt_preview": attack_prompt[:150],
        "cpb_query_risk":       _safe(retrieval["query_risk"].score),
        "cpb_query_pii_score":  _safe(retrieval.get("query_pii_score")),
        "n_raw_chunks":         len(raw_chunks),
        "n_masked_chunks_sent": len(masked_chunks),
    }

    if retrieval.get("decision") == "direct_suppression":
        return {**base,
            "raw_response":    "",
            "predicted_member": False,
            "cpb_decision":    "direct_suppression",
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
            "tokens_prompt":    0,
            "tokens_completion": 0,
        }

    if not masked_chunks:
        return {**base,
            "raw_response":    "",
            "predicted_member": False,
            "cpb_decision":    "all_chunks_suppressed",
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
            "tokens_prompt":    0,
            "tokens_completion": 0,
        }

    # --- Bloc 5 : Génération LLM ---
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

    predicted_member = _parse_response(response)
    pii_res          = compute_pii_leakage(response, raw_chunks)

    audit           = retrieval.get("audit")
    chunk_decisions = retrieval.get("chunk_decisions", [])
    n_masked_c, n_suppressed_c = _cpb_chunk_counts(chunk_decisions)

    return {**base,
        "raw_response":    response,
        "predicted_member": predicted_member,
        "cpb_decision":    "retrieval_masked",
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
        "tokens_prompt":    llm_resp.tokens_prompt,
        "tokens_completion": llm_resp.tokens_completion,
    }


# ---------------------------------------------------------------------------
# Log MLflow par sample
# ---------------------------------------------------------------------------

def log_sample(result: dict, elapsed_s: float) -> None:
    run_name = f"{LLM_NAME}__{RAG_NAME}__mia__{result['sample_id']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("llm",              LLM_NAME)
        mlflow.log_param("rag_architecture", RAG_NAME)
        mlflow.log_param("attack",           ATTACK_NAME)
        mlflow.log_param("sample_id",        result["sample_id"])
        mlflow.log_param("is_member",        str(result["is_member"]))
        mlflow.log_param("predicted_member", str(result["predicted_member"]))
        mlflow.log_param("cpb_decision",     result["cpb_decision"])
        mlflow.log_param("cpb_sad_decision", result["cpb_sad_decision"])
        mlflow.log_param("cpb_response_guard_decision", result["cpb_response_guard_decision"])
        mlflow.log_param("cpb_sad_categories",
            ",".join(result["cpb_sad_categories"]) if result["cpb_sad_categories"] else "")
        mlflow.log_param("text_preview",     result["text_preview"])
        mlflow.log_param("response_preview", result["raw_response"][:200])

        mlflow.log_metric("is_member",         int(result["is_member"]))
        mlflow.log_metric("predicted_member",  int(result["predicted_member"]))
        mlflow.log_metric("correct",           int(result["is_member"] == result["predicted_member"]))
        mlflow.log_metric("cpb_query_risk",    result["cpb_query_risk"])
        mlflow.log_metric("cpb_query_pii_score", result["cpb_query_pii_score"])
        mlflow.log_metric("cpb_n_masked_chunks", result["cpb_n_masked_chunks"])
        mlflow.log_metric("cpb_n_suppressed_chunks", result["cpb_n_suppressed_chunks"])
        mlflow.log_metric("cpb_max_pii_score", result["cpb_max_pii_score"])
        mlflow.log_metric("cpb_min_budget",    result["cpb_min_budget"])
        mlflow.log_metric("cpb_sad_detected",  int(result["cpb_sad_detected"]))
        mlflow.log_metric("cpb_leakage_score", result["cpb_leakage_score"])
        mlflow.log_metric("pii_leakage_rate",  result["pii_leakage_rate"])
        mlflow.log_metric("n_pii_total",       result["n_pii_total"])
        mlflow.log_metric("n_pii_leaked",      result["n_pii_leaked"])
        mlflow.log_metric("n_raw_chunks",      result["n_raw_chunks"])
        mlflow.log_metric("n_masked_chunks_sent", result["n_masked_chunks_sent"])
        mlflow.log_metric("tokens_prompt",     result["tokens_prompt"])
        mlflow.log_metric("tokens_completion", result["tokens_completion"])
        mlflow.log_metric("tokens_total",      result["tokens_prompt"] + result["tokens_completion"])
        mlflow.log_metric("response_time_s",   elapsed_s)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def get_done_sample_ids() -> set[str]:
    """Return sample_ids already logged as FINISHED runs in this experiment."""
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
        sid = r.data.params.get("sample_id")
        if sid:
            done.add(sid)
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    done_ids = get_done_sample_ids()
    if done_ids:
        print(f"   ⏩ {len(done_ids)} samples déjà traités — reprise")

    print("📦 Connexion à ChromaDB...")
    store = ChromaStore()

    print(f"📥 Chargement des members ({N_PAIRS} chunks ChromaDB)...")
    members = load_member_samples(store, N_PAIRS)
    print(f"   {len(members)} members chargés")

    print(f"📥 Chargement des non-members ({N_PAIRS} textes)...")
    non_member_texts = load_non_member_texts(N_PAIRS)
    print(f"   {len(non_member_texts)} non-members chargés")

    print("🤖 Chargement Llama 3.1 8B...")
    llm = LlamaLLM()

    print("🛡️  Initialisation CPBNaiveRAG...")
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb_rag   = CPBNaiveRAG(naive_rag=naive_rag, architecture_name=RAG_NAME)

    # Assemblage : members + non-members avec labels
    samples = (
        [(f"member_{i:03d}",     text["text"], True)  for i, text in enumerate(members)]
        + [(f"nonmember_{i:03d}", text,         False) for i, text in enumerate(non_member_texts)]
    )
    random.shuffle(samples)

    all_results: list[dict] = []

    print(f"\n{'='*65}")
    print(f"  MIA Attack — {LLM_NAME} × {RAG_NAME}  ({len(samples)} samples)")
    print(f"  Prompt variant : #{PROMPT_ID}")
    print(f"  Expérience MLflow : {EXPERIMENT_NAME}")
    print(f"{'='*65}\n")

    for sample_id, text, is_member in samples:
        if sample_id in done_ids:
            print(f"  [{sample_id}] skip (déjà traité)")
            continue

        label = "member    " if is_member else "non-member"
        print(f"  [{sample_id}] {label} | {text[:60]!r}")

        t0 = time.perf_counter()
        result = run_sample(text, is_member, sample_id, cpb_rag)
        elapsed_s = time.perf_counter() - t0

        log_sample(result, elapsed_s)
        all_results.append(result)

        cpb_blocked = result["cpb_decision"] in ("direct_suppression", "all_chunks_suppressed")
        pred_str = "YES" if result["predicted_member"] else "NO "
        correct  = "✅" if result["is_member"] == result["predicted_member"] else "❌"
        status   = "🛡️ " if cpb_blocked else correct

        print(
            f"     {status} pred={pred_str} | "
            f"risk={result['cpb_query_risk']:.2f} | "
            f"sad={int(result['cpb_sad_detected'])} | "
            f"{elapsed_s:.1f}s"
        )

    # --- Métriques agrégées ---
    if not all_results:
        print("\n⚠️  Aucun résultat à agréger.")
        return

    y_true = np.array([int(r["is_member"])        for r in all_results])
    y_pred = np.array([int(r["predicted_member"])  for r in all_results])

    members_res     = [r for r in all_results if r["is_member"]]
    non_members_res = [r for r in all_results if not r["is_member"]]

    tpr = sum(r["predicted_member"] for r in members_res)     / len(members_res)     if members_res     else 0.0
    fpr = sum(r["predicted_member"] for r in non_members_res) / len(non_members_res) if non_members_res else 0.0

    try:
        from sklearn.metrics import roc_auc_score
        auc_roc = float(roc_auc_score(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan")
    except Exception:
        auc_roc = float("nan")

    n_cpb_blocked = sum(
        1 for r in all_results
        if r["cpb_decision"] in ("direct_suppression", "all_chunks_suppressed")
    )
    n_correct    = int((y_true == y_pred).sum())
    accuracy     = n_correct / len(all_results)
    pii_mean     = sum(r["pii_leakage_rate"] for r in all_results) / len(all_results)

    with mlflow.start_run(run_name=f"{LLM_NAME}__{RAG_NAME}__mia__SUMMARY"):
        mlflow.log_param("llm",              LLM_NAME)
        mlflow.log_param("rag_architecture", RAG_NAME)
        mlflow.log_param("attack",           ATTACK_NAME)
        mlflow.log_param("n_pairs",          N_PAIRS)
        mlflow.log_param("prompt_id",        PROMPT_ID)
        mlflow.log_metric("tpr",             tpr)
        mlflow.log_metric("fpr",             fpr)
        mlflow.log_metric("auc_roc",         auc_roc if not np.isnan(auc_roc) else -1.0)
        mlflow.log_metric("accuracy",        accuracy)
        mlflow.log_metric("pii_leakage_mean", pii_mean)
        mlflow.log_metric("n_cpb_blocked",   n_cpb_blocked)
        mlflow.log_metric("n_total",         len(all_results))

    print(f"\n{'='*65}")
    print(f"  RÉSUMÉ — {LLM_NAME} × {RAG_NAME}")
    print(f"{'='*65}")
    print(f"  TPR (Sensitivity)    : {tpr:.4f}  (membres prédits membres)")
    print(f"  FPR (False Positive) : {fpr:.4f}  (non-membres prédits membres)")
    print(f"  AUC-ROC              : {auc_roc:.4f}" if not np.isnan(auc_roc) else "  AUC-ROC              : N/A")
    print(f"  Accuracy             : {accuracy:.4f}")
    print(f"  Bloqués par CPB      : {n_cpb_blocked}/{len(all_results)}")
    print(f"\n  MLflow → lance 'mlflow ui' et ouvre '{EXPERIMENT_NAME}'")


if __name__ == "__main__":
    main()
