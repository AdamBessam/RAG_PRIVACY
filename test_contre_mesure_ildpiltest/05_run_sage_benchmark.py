"""
Étape 5 — Benchmark SAGE RAG sur les 1000 queries.

Lance les mêmes 1000 queries que le benchmark NaiveRAG vs CPB (03_run_benchmark.py)
mais en utilisant le SAGE RAG (NaiveRAG sur la base de données synthétiques).

Métriques mesurées :
  query_id, query_type, query
  sage_response, sage_pii_leaked, sage_pii_total, sage_pii_rate, sage_rouge_l, sage_latency_s

PII leakage SAGE (métrique corrigée) :
  - Utilise doc_pii_surviving.json (généré par 05a_annotate_pii_gt.py)
  - Pour chaque chunk récupéré, on connaît les PII originales qui ont survécu dans le synthétique
  - sage_pii_total = nb de PII survivantes dans les chunks récupérés
  - sage_pii_leaked = nb de ces PII survivantes présentes dans la réponse SAGE
  - Mesure réelle : est-ce que SAGE laisse fuir les PII qu'il n'a pas su anonymiser ?

  Prérequis : python test_contre_mesure_ildpiltest/05a_annotate_pii_gt.py

Usage :
    python test_contre_mesure_ildpiltest/05_run_sage_benchmark.py
    python test_contre_mesure_ildpiltest/05_run_sage_benchmark.py --llm llama       # défaut
    python test_contre_mesure_ildpiltest/05_run_sage_benchmark.py --llm mistral
    python test_contre_mesure_ildpiltest/05_run_sage_benchmark.py --llm gpt4o-mini
    python test_contre_mesure_ildpiltest/05_run_sage_benchmark.py --limit 50        # test rapide
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
    QUERIES_FILE,
    MLFLOW_DIR, MLFLOW_EXPERIMENT,
    TOP_K,
)
from test_contre_mesure_ildpiltest._store import IldpilTestStore
from rag.naive_rag import NaiveRAG

SAGE_CHROMA_DIR       = str(Path(__file__).parent / "chroma_db_sage")
SAGE_COLLECTION_NAME  = "ildpil_sage_synthetic"
RESULTS_CSV           = Path(__file__).parent / "sage_benchmark_results.csv"
CHECKPOINT_FILE       = Path(__file__).parent / "sage_checkpoint_run.json"
DOC_PII_SURVIVING_FILE = Path(__file__).parent / "doc_pii_surviving.json"


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

def measure_pii_leakage_surviving(
    response: str,
    chunks: list[dict],
    pii_surviving: dict[str, list[str]],
) -> tuple[int, int]:
    """
    Mesure la fuite PII réelle pour SAGE.
    Ground truth = PII originales qui ont survécu dans le texte synthétique
    (celles que SAGE n'a pas su anonymiser), issues de doc_pii_surviving.json.
    """
    if not response or not chunks:
        return 0, 0
    pii_texts = set()
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "")
        for pii in pii_surviving.get(doc_id, []):
            if pii and len(pii) > 2:
                pii_texts.add(pii.lower())
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

def load_checkpoint() -> list[dict]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Checkpoint trouvé : {len(data)} queries déjà traitées — reprise")
        return data
    return []


def save_checkpoint(results: list[dict]):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(queries: list[dict], sage_rag: NaiveRAG, pii_surviving: dict) -> list[dict]:
    results   = load_checkpoint()
    done_ids  = {r["query_id"] for r in results}
    remaining = [q for q in queries if q.get("global_id", q["query_id"]) not in done_ids]

    if not remaining:
        print("Toutes les queries sont déjà traitées (checkpoint complet).")
        return results

    print(f"{len(remaining)} queries restantes sur {len(queries)} total\n")

    for q in tqdm(remaining, desc="Benchmark SAGE RAG"):
        query_text = q["query"]
        if not isinstance(query_text, str):
            query_text = str(query_text)
        query_id   = q.get("global_id", q["query_id"])
        query_type = q["query_type"]

        t0 = time.time()
        try:
            out        = sage_rag.run(query_text, top_k=TOP_K)
            sage_resp  = out.get("response", "")
            sage_chunks = out.get("chunks", [])
        except Exception as exc:
            sage_resp   = f"ERROR: {exc}"
            sage_chunks = []

        latency = round(time.time() - t0, 3)

        pii_leaked, pii_total = measure_pii_leakage_surviving(sage_resp, sage_chunks, pii_surviving)
        pii_rate = round(pii_leaked / pii_total, 4) if pii_total > 0 else 0.0
        rouge    = measure_rouge_l(sage_resp, sage_chunks)

        results.append({
            "query_id":        query_id,
            "query_type":      query_type,
            "query":           query_text,
            "sage_response":   sage_resp,
            "sage_pii_leaked": pii_leaked,
            "sage_pii_total":  pii_total,
            "sage_pii_rate":   pii_rate,
            "sage_rouge_l":    rouge,
            "sage_latency_s":  latency,
        })
        save_checkpoint(results)

    CHECKPOINT_FILE.unlink(missing_ok=True)
    return results


# ── MLflow ────────────────────────────────────────────────────────────────────

def log_to_mlflow(results: list[dict], llm_name: str):
    mlflow.set_tracking_uri(Path(MLFLOW_DIR).resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"SAGE_RAG_{llm_name}"
    with mlflow.start_run(run_name=run_name):
        total = len(results)
        mlflow.log_param("llm",            llm_name)
        mlflow.log_param("n_queries",      total)
        mlflow.log_param("dataset",        "ildpil/text-anonymization-benchmark")
        mlflow.log_param("split",          "test")
        mlflow.log_param("protect_method", "SAGE_agent2")
        mlflow.log_param("stage1_llm",     "gpt-3.5-turbo")
        mlflow.log_param("stage2_llm",     "gpt-4")

        sage_leaked = sum(r["sage_pii_leaked"] for r in results)
        sage_total  = sum(r["sage_pii_total"]  for r in results)
        sage_pii    = sage_leaked / sage_total if sage_total > 0 else 0.0
        sage_rl     = sum(r["sage_rouge_l"]    for r in results) / total
        sage_lat    = sum(r["sage_latency_s"]  for r in results) / total

        mlflow.log_metric("sage_pii_leaked_total", sage_leaked)
        mlflow.log_metric("sage_pii_total",        sage_total)
        mlflow.log_metric("sage_pii_leakage_rate", round(sage_pii, 4))
        mlflow.log_metric("sage_rouge_l_mean",     round(sage_rl,  4))
        mlflow.log_metric("sage_latency_mean_s",   round(sage_lat, 3))

        # Métriques par type de query
        query_types = sorted(set(r["query_type"] for r in results))
        for qtype in query_types:
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            if n == 0:
                continue
            s_leaked = sum(r["sage_pii_leaked"] for r in subset)
            s_total  = sum(r["sage_pii_total"]  for r in subset)
            mlflow.log_metric(f"{qtype}_sage_pii_rate",  round(s_leaked / s_total, 4) if s_total > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_sage_rouge_l",   round(sum(r["sage_rouge_l"] for r in subset) / n, 4))
            mlflow.log_metric(f"{qtype}_n_queries",      n)

        # CSV
        fieldnames = [
            "query_id", "query_type", "query",
            "sage_response", "sage_pii_leaked", "sage_pii_total",
            "sage_pii_rate", "sage_rouge_l", "sage_latency_s",
        ]
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        mlflow.log_artifact(str(RESULTS_CSV), artifact_path="results")
        print(f"\nCSV des résultats : {RESULTS_CSV}")

    print(f"MLflow experiment : {MLFLOW_EXPERIMENT}")
    print(f"MLflow tracking   : {MLFLOW_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",   default="llama",
                        choices=["llama", "mistral", "gpt4o-mini", "claude-haiku"],
                        help="LLM de génération RAG (papier : llama3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite le nombre de queries (test rapide)")
    args = parser.parse_args()

    if not QUERIES_FILE.exists():
        print(f"ERREUR : {QUERIES_FILE} introuvable.")
        print("Lancez d'abord : python test_contre_mesure_ildpiltest/02_generate_queries.py")
        sys.exit(1)

    if not DOC_PII_SURVIVING_FILE.exists():
        print(f"ERREUR : {DOC_PII_SURVIVING_FILE} introuvable.")
        print("Lancez d'abord : python test_contre_mesure_ildpiltest/05a_annotate_pii_gt.py")
        sys.exit(1)

    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries = json.load(f)

    with open(DOC_PII_SURVIVING_FILE, encoding="utf-8") as f:
        pii_surviving = json.load(f)
    n_docs_with_leak = sum(1 for v in pii_surviving.values() if v)
    print(f"doc_pii_surviving.json chargé : {len(pii_surviving)} docs, "
          f"{n_docs_with_leak} avec PII survivantes")

    if args.limit:
        queries = queries[:args.limit]
        print(f"Mode test : {args.limit} queries seulement")

    print(f"{len(queries)} queries chargées")

    print(f"\nInitialisation ChromaDB SAGE ({SAGE_CHROMA_DIR})...")
    store = IldpilTestStore(chroma_dir=SAGE_CHROMA_DIR, collection_name=SAGE_COLLECTION_NAME)
    if store.count() == 0:
        print("ERREUR : collection SAGE vide.")
        print("Lancez d'abord : python test_contre_mesure_ildpiltest/04_index_sage.py")
        sys.exit(1)

    print(f"Initialisation LLM : {args.llm}...")
    llm      = build_llm(args.llm)
    sage_rag = NaiveRAG(store=store, llm=llm)

    print(f"\nDémarrage du benchmark SAGE ({len(queries)} queries) — LLM : {args.llm}...\n")
    results = run_benchmark(queries, sage_rag, pii_surviving)

    # Sauvegarde CSV immédiate — indépendante de MLflow
    fieldnames = [
        "query_id", "query_type", "query",
        "sage_response", "sage_pii_leaked", "sage_pii_total",
        "sage_pii_rate", "sage_rouge_l", "sage_latency_s",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV sauvegardé : {RESULTS_CSV}")

    print(f"\nLogging dans MLflow ({MLFLOW_DIR})...")
    log_to_mlflow(results, args.llm)

    # Résumé console
    total       = len(results)
    sage_leaked = sum(r["sage_pii_leaked"] for r in results)
    sage_total  = sum(r["sage_pii_total"]  for r in results)
    sage_pii    = sage_leaked / sage_total if sage_total > 0 else 0.0
    sage_rl     = sum(r["sage_rouge_l"]    for r in results) / total
    sage_lat    = sum(r["sage_latency_s"]  for r in results) / total

    print(f"\n{'='*55}")
    print(f"  RÉSULTATS SAGE RAG — {total} queries")
    print(f"{'='*55}")
    print(f"  {'Métrique':<35} {'SAGE':>10}")
    print(f"  {'-'*50}")
    print(f"  {'PII leakage rate':<35} {sage_pii:>10.1%}")
    print(f"  {'PII leaked / total':<35} {sage_leaked}/{sage_total}")
    print(f"  {'ROUGE-L moyen':<35} {sage_rl:>10.4f}")
    print(f"  {'Latence moyenne (s)':<35} {sage_lat:>10.3f}")
    print(f"{'='*55}")
    print(f"\n  Résultats complets : {RESULTS_CSV}")
    print(f"\n  NOTE : PII leakage mesuré sur les PII originales survivantes")
    print(f"         dans les textes synthétiques (doc_pii_surviving.json).")

    # Auto-push
    print("\nPush automatique des résultats sur GitHub...")
    try:
        subprocess.run(["git", "add", str(RESULTS_CSV)], check=True)
        subprocess.run(["git", "commit", "-m",
                        f"auto: SAGE benchmark results ildpil test ({args.llm}, {total} queries)"],
                       check=True)
        subprocess.run(["git", "push"], check=True)
        print("Résultats pushés sur GitHub avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Push automatique échoué : {e}")


if __name__ == "__main__":
    main()
