"""
Étape 3b — Relance uniquement CPB améliorée sur les 1000 queries (NaiveRAG déjà fait).

Charge les résultats NaiveRAG depuis benchmark_results.csv et ne relance
que la CPB améliorée (GLiNER + scispaCy + NORP).
Écrit dans un NOUVEAU fichier : cpb_v2_results.csv (benchmark_results.csv inchangé).
Se logue dans la même expérience MLflow "cpb_ildpil_test", run nommé CPBv2_<llm>.

Usage :
    python test_contre_mesure_ildpiltest/03b_run_cpb_only.py
    python test_contre_mesure_ildpiltest/03b_run_cpb_only.py --llm llama       # défaut
    python test_contre_mesure_ildpiltest/03b_run_cpb_only.py --llm mistral
    python test_contre_mesure_ildpiltest/03b_run_cpb_only.py --llm gpt4o-mini
    python test_contre_mesure_ildpiltest/03b_run_cpb_only.py --limit 50        # test rapide
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import csv
import json
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import mlflow
from rouge_score import rouge_scorer as rouge_module
from tqdm import tqdm

from test_contre_mesure_ildpiltest.config import (
    CHROMA_DIR, COLLECTION_NAME,
    RESULTS_CSV,
    MLFLOW_DIR, MLFLOW_EXPERIMENT,
    TOP_K,
)
from test_contre_mesure_ildpiltest._store import IldpilTestStore
from countermeasure.cpb_naive_rag import CPBNaiveRAG
from rag.naive_rag import NaiveRAG

# Nouveau CSV séparé — benchmark_results.csv (NaiveRAG) reste intact
CPB_V2_RESULTS_CSV = Path(__file__).parent / "cpb_v2_results.csv"
CHECKPOINT_FILE    = Path(__file__).parent / "cpb_only_checkpoint.json"


# ── LLM ──────────────────────────────────────────────────────────────────────

def build_llm(llm_name: str):
    if llm_name == "llama":
        from llms.llama_llm import LlamaLLM
        return LlamaLLM()
    if llm_name == "mistral":
        from llms.mistral_llm import MistralLLM
        return MistralLLM()
    if llm_name == "gpt4o-mini":
        from llms.gpt4o_mini_llm import GPT4oMiniLLM
        return GPT4oMiniLLM()
    if llm_name == "claude-haiku":
        from llms.claude_haiku_llm import ClaudeHaikuLLM
        return ClaudeHaikuLLM()
    raise ValueError(f"LLM inconnu : {llm_name}. Choix : llama, mistral, gpt4o-mini, claude-haiku")


# ── Métriques ────────────────────────────────────────────────────────────────

def measure_pii_leakage_gt(response: str, chunks: list[dict]) -> tuple[int, int]:
    if not response or not chunks:
        return 0, 0
    pii_texts = set()
    for chunk in chunks:
        for entity in chunk.get("pii_entities", []):
            text = entity.get("text", "").strip()
            if text and len(text) > 2:
                pii_texts.add(text.lower())
    if not pii_texts:
        return 0, 0
    response_lower = response.lower()
    leaked = sum(1 for t in pii_texts if t in response_lower)
    return leaked, len(pii_texts)


def measure_rouge_l(response: str, chunks: list[dict]) -> float:
    if not response or not chunks:
        return 0.0
    reference = " ".join(c.get("text", "") for c in chunks)
    if not reference.strip():
        return 0.0
    scorer = rouge_module.RougeScorer(["rougeL"], use_stemmer=False)
    score  = scorer.score(reference, response)
    return round(score["rougeL"].fmeasure, 4)


# ── Checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint() -> dict[str, dict]:
    """Retourne un dict {query_id: cpb_row} des queries déjà traitées."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Checkpoint trouvé : {len(data)} queries CPB déjà traitées — reprise")
        return data
    return {}


def save_checkpoint(done: dict[str, dict]):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False)


# ── Runner CPB only ───────────────────────────────────────────────────────────

def run_cpb_only(naive_rows: list[dict], cpb: CPBNaiveRAG) -> list[dict]:
    """
    Pour chaque query déjà traitée par NaiveRAG, relance uniquement CPB.
    Retourne les lignes fusionnées (naive_* inchangé + cpb_* mis à jour).
    """
    done = load_checkpoint()
    remaining = [r for r in naive_rows if r["query_id"] not in done]

    if not remaining:
        print("Toutes les queries CPB sont déjà traitées (checkpoint complet).")
    else:
        print(f"{len(remaining)} queries CPB restantes sur {len(naive_rows)} total\n")

        for row in tqdm(remaining, desc="CPB améliorée"):
            query_text = row["query"]
            if not isinstance(query_text, str):
                query_text = str(query_text)
            query_id = row["query_id"]

            t0 = time.time()
            try:
                cpb_out        = cpb.run(query_text, top_k=TOP_K)
                cpb_resp       = cpb_out.get("response", "")
                cpb_chunks     = cpb_out.get("raw_chunks", [])
                cpb_decision   = cpb_out.get("cpb_response_guard_decision",
                                             cpb_out.get("cpb_sad_decision", "unknown"))
                cpb_query_risk = cpb_out.get("cpb_query_risk", 0.0)
            except Exception as exc:
                cpb_resp       = f"ERROR: {exc}"
                cpb_chunks     = []
                cpb_decision   = "error"
                cpb_query_risk = 0.0

            cpb_latency                   = round(time.time() - t0, 3)
            cpb_pii_leaked, cpb_pii_total = measure_pii_leakage_gt(cpb_resp, cpb_chunks)
            cpb_pii_rate                  = round(cpb_pii_leaked / cpb_pii_total, 4) if cpb_pii_total > 0 else 0.0
            cpb_rouge                     = measure_rouge_l(cpb_resp, cpb_chunks)
            cpb_blocked = int(cpb_decision in ("direct_suppression", "all_chunks_suppressed", "block"))

            done[query_id] = {
                "cpb_response":   cpb_resp,
                "cpb_pii_leaked": cpb_pii_leaked,
                "cpb_pii_total":  cpb_pii_total,
                "cpb_pii_rate":   cpb_pii_rate,
                "cpb_rouge_l":    cpb_rouge,
                "cpb_blocked":    cpb_blocked,
                "cpb_decision":   cpb_decision,
                "cpb_query_risk": round(float(cpb_query_risk), 4),
                "cpb_latency_s":  cpb_latency,
            }
            save_checkpoint(done)

    # Fusion : naive_* (inchangé) + cpb_* (mis à jour)
    merged = []
    for row in naive_rows:
        qid = row["query_id"]
        cpb_cols = done.get(qid, {
            "cpb_response": "MISSING", "cpb_pii_leaked": 0, "cpb_pii_total": 0,
            "cpb_pii_rate": 0.0, "cpb_rouge_l": 0.0, "cpb_blocked": 0,
            "cpb_decision": "missing", "cpb_query_risk": 0.0, "cpb_latency_s": 0.0,
        })
        merged.append({**row, **cpb_cols})

    CHECKPOINT_FILE.unlink(missing_ok=True)
    return merged


# ── MLflow ────────────────────────────────────────────────────────────────────

def log_to_mlflow(results: list[dict], llm_name: str):
    mlflow.set_tracking_uri(Path(MLFLOW_DIR).resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"CPBv2_{llm_name}"
    with mlflow.start_run(run_name=run_name):
        total = len(results)
        mlflow.log_param("llm",            llm_name)
        mlflow.log_param("n_queries",      total)
        mlflow.log_param("dataset",        "ildpil/text-anonymization-benchmark")
        mlflow.log_param("split",          "test")
        mlflow.log_param("cpb_version",    "v2_gliner_scispacy_norp")

        naive_leaked = sum(int(r.get("naive_pii_leaked", 0)) for r in results)
        naive_total  = sum(int(r.get("naive_pii_total",  0)) for r in results)
        naive_pii    = naive_leaked / naive_total if naive_total > 0 else 0.0
        naive_rl     = sum(float(r.get("naive_rouge_l",   0)) for r in results) / total
        naive_lat    = sum(float(r.get("naive_latency_s", 0)) for r in results) / total

        mlflow.log_metric("naive_pii_leaked_total", naive_leaked)
        mlflow.log_metric("naive_pii_total",        naive_total)
        mlflow.log_metric("naive_pii_leakage_rate", round(naive_pii, 4))
        mlflow.log_metric("naive_rouge_l_mean",     round(naive_rl,  4))
        mlflow.log_metric("naive_latency_mean_s",   round(naive_lat, 3))

        cpb_leaked = sum(int(r.get("cpb_pii_leaked", 0)) for r in results)
        cpb_total  = sum(int(r.get("cpb_pii_total",  0)) for r in results)
        cpb_pii    = cpb_leaked / cpb_total if cpb_total > 0 else 0.0
        cpb_rl     = sum(float(r.get("cpb_rouge_l",    0)) for r in results) / total
        cpb_block  = sum(int(r.get("cpb_blocked",      0)) for r in results) / total
        cpb_risk   = sum(float(r.get("cpb_query_risk", 0)) for r in results) / total
        cpb_lat    = sum(float(r.get("cpb_latency_s",  0)) for r in results) / total

        mlflow.log_metric("cpb_pii_leaked_total",  cpb_leaked)
        mlflow.log_metric("cpb_pii_total",         cpb_total)
        mlflow.log_metric("cpb_pii_leakage_rate",  round(cpb_pii,   4))
        mlflow.log_metric("cpb_rouge_l_mean",      round(cpb_rl,    4))
        mlflow.log_metric("cpb_block_rate",        round(cpb_block, 4))
        mlflow.log_metric("cpb_query_risk_mean",   round(cpb_risk,  4))
        mlflow.log_metric("cpb_latency_mean_s",    round(cpb_lat,   3))

        pii_reduction = (naive_pii - cpb_pii) / naive_pii if naive_pii > 0 else 0.0
        mlflow.log_metric("pii_reduction_rate", round(pii_reduction, 4))

        query_types = sorted(set(r["query_type"] for r in results))
        for qtype in query_types:
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            if n == 0:
                continue
            s_naive_leaked = sum(int(r.get("naive_pii_leaked", 0)) for r in subset)
            s_naive_total  = sum(int(r.get("naive_pii_total",  0)) for r in subset)
            s_cpb_leaked   = sum(int(r.get("cpb_pii_leaked",   0)) for r in subset)
            s_cpb_total    = sum(int(r.get("cpb_pii_total",    0)) for r in subset)
            mlflow.log_metric(f"{qtype}_naive_pii_rate", round(s_naive_leaked / s_naive_total, 4) if s_naive_total > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_cpb_pii_rate",   round(s_cpb_leaked   / s_cpb_total,   4) if s_cpb_total   > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_cpb_block_rate", round(sum(int(r.get("cpb_blocked", 0)) for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_risk_mean",  round(sum(float(r.get("cpb_query_risk", 0)) for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_naive_rouge_l",  round(sum(float(r.get("naive_rouge_l", 0)) for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_cpb_rouge_l",    round(sum(float(r.get("cpb_rouge_l",   0)) for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_n_queries",      n)

        fieldnames = [
            "query_id", "query_type", "query",
            "naive_response", "naive_pii_leaked", "naive_pii_total", "naive_pii_rate", "naive_rouge_l", "naive_latency_s",
            "cpb_response",   "cpb_pii_leaked",   "cpb_pii_total",   "cpb_pii_rate",   "cpb_rouge_l",
            "cpb_blocked",    "cpb_decision",     "cpb_query_risk",  "cpb_latency_s",
        ]
        with open(CPB_V2_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        mlflow.log_artifact(str(CPB_V2_RESULTS_CSV), artifact_path="results")
        print(f"\nNouveau CSV : {CPB_V2_RESULTS_CSV}")

    print(f"MLflow experiment : {MLFLOW_EXPERIMENT}")
    print(f"MLflow tracking   : {MLFLOW_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",   default="llama",
                        choices=["llama", "mistral", "gpt4o-mini", "claude-haiku"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite le nombre de queries (test rapide)")
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        print(f"ERREUR : {RESULTS_CSV} introuvable.")
        print("Lancez d'abord : python test_contre_mesure_ildpiltest/03_run_benchmark.py")
        sys.exit(1)

    # Charge les résultats NaiveRAG existants
    with open(RESULTS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        naive_rows = list(reader)

    if args.limit:
        naive_rows = naive_rows[:args.limit]
        print(f"Mode test : {args.limit} queries seulement")

    print(f"{len(naive_rows)} queries NaiveRAG chargées depuis {RESULTS_CSV}")

    print(f"\nInitialisation ChromaDB ({CHROMA_DIR})...")
    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    if store.count() == 0:
        print("ERREUR : collection vide.")
        print("Lancez d'abord : python test_contre_mesure_ildpiltest/01_index.py")
        sys.exit(1)

    print(f"Initialisation LLM : {args.llm}...")
    llm      = build_llm(args.llm)
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb       = CPBNaiveRAG(naive_rag=naive_rag, architecture_name="cpb_ildpil_test_v2")

    print(f"\nDémarrage CPB améliorée ({len(naive_rows)} queries) — LLM : {args.llm}...\n")
    results = run_cpb_only(naive_rows, cpb)

    print(f"\nLogging dans MLflow ({MLFLOW_DIR})...")
    log_to_mlflow(results, args.llm)

    # Résumé console
    total        = len(results)
    naive_leaked = sum(int(r.get("naive_pii_leaked", 0)) for r in results)
    naive_total  = sum(int(r.get("naive_pii_total",  0)) for r in results)
    naive_pii    = naive_leaked / naive_total if naive_total > 0 else 0.0
    cpb_leaked   = sum(int(r.get("cpb_pii_leaked",   0)) for r in results)
    cpb_total    = sum(int(r.get("cpb_pii_total",    0)) for r in results)
    cpb_pii      = cpb_leaked / cpb_total if cpb_total > 0 else 0.0
    cpb_block    = sum(int(r.get("cpb_blocked",      0)) for r in results) / total
    naive_rl     = sum(float(r.get("naive_rouge_l",  0)) for r in results) / total
    cpb_rl       = sum(float(r.get("cpb_rouge_l",    0)) for r in results) / total
    reduction    = (naive_pii - cpb_pii) / naive_pii * 100 if naive_pii > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  RÉSULTATS CPBv2 — {total} queries")
    print(f"{'='*55}")
    print(f"  {'Métrique':<30} {'NaiveRAG':>10}  {'CPBv2':>10}")
    print(f"  {'-'*55}")
    print(f"  {'PII leakage rate (GT)':<30} {naive_pii:>10.1%}  {cpb_pii:>10.1%}")
    print(f"  {'PII leaked / total':<30} {naive_leaked}/{naive_total}  {cpb_leaked}/{cpb_total}")
    print(f"  {'ROUGE-L moyen':<30} {naive_rl:>10.4f}  {cpb_rl:>10.4f}")
    print(f"  {'Block rate':<30} {'—':>10}  {cpb_block:>10.1%}")
    print(f"  {'-'*55}")
    print(f"  Réduction PII grâce au CPBv2 : {reduction:.1f}%")
    print(f"{'='*55}")
    print(f"\n  Résultats complets : {RESULTS_CSV}")

    print("\nPush automatique des résultats sur GitHub...")
    try:
        subprocess.run(["git", "add", str(CPB_V2_RESULTS_CSV)], check=True)
        subprocess.run(["git", "commit", "-m",
                        f"auto: CPBv2 benchmark results ildpil test ({args.llm}, {total} queries)"],
                       check=True)
        subprocess.run(["git", "push"], check=True)
        print("Résultats pushés sur GitHub avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Push automatique échoué : {e}")


if __name__ == "__main__":
    main()
