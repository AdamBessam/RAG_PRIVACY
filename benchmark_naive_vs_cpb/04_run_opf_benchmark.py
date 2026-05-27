"""
Étape 4 — Benchmark OPFNaiveRAG (OpenAI Privacy Filter)
========================================================
Les résultats NaiveRAG sont déjà dans benchmark_results.csv (étape 02).
Ce script ne les recalcule PAS — il les relit depuis le CSV.

Ce script lance uniquement :
  OPFNaiveRAG — retrieve top-k → LLM génère la réponse sur contexte COMPLET
                               → OPF redacte les PII dans la RÉPONSE du LLM

Pipeline post-génération : le LLM voit tout, OPF filtre ce qui sort.
Puis fusionne avec les résultats NaiveRAG existants pour le logging MLflow.

Métriques loggées (CSV + MLflow) :
  - PII leakage rate OPF (ground-truth)
  - Nb entités redactées par OPF / chunk
  - Réduction PII vs NaiveRAG (depuis CSV existant)

Reprise automatique depuis le checkpoint si le script crash.

Usage:
    python benchmark_naive_vs_cpb/04_run_opf_benchmark.py
    python benchmark_naive_vs_cpb/04_run_opf_benchmark.py --llm llama        (défaut)
    python benchmark_naive_vs_cpb/04_run_opf_benchmark.py --llm gpt4o-mini
    python benchmark_naive_vs_cpb/04_run_opf_benchmark.py --llm mistral
    python benchmark_naive_vs_cpb/04_run_opf_benchmark.py --llm claude-haiku
    python benchmark_naive_vs_cpb/04_run_opf_benchmark.py --limit 5          (test rapide)
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import mlflow
from tqdm import tqdm

from benchmark_naive_vs_cpb.config import (
    CHROMA_DIR, COLLECTION_NAME,
    QUERIES_FILE, RESULTS_CSV,
    MLFLOW_DIR, MLFLOW_EXPERIMENT,
    TOP_K,
)
from benchmark_naive_vs_cpb._store import BenchmarkStore
from rag.naive_rag import NaiveRAG
from countermeasure.opf_naive_rag import OPFNaiveRAG

BENCHMARK_DIR  = Path(__file__).parent
OPF_RESULTS_CSV = BENCHMARK_DIR / "opf_benchmark_results.csv"
OPF_CHECKPOINT  = BENCHMARK_DIR / "opf_checkpoint.json"


# ── LLM factory ──────────────────────────────────────────────────────────────

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
    raise ValueError(
        f"LLM inconnu : {llm_name}. Choix : llama, mistral, gpt4o-mini, claude-haiku"
    )


# ── Chargement des résultats NaiveRAG existants ───────────────────────────────

def load_naive_results(path: Path) -> dict[str, dict]:
    """
    Lit benchmark_results.csv et retourne un dict {query_id: row}.
    On ne garde que les colonnes NaiveRAG nécessaires.
    """
    naive_by_id = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            naive_by_id[qid] = {
                "naive_response":   row.get("naive_response", ""),
                "naive_pii_leaked": int(float(row.get("naive_pii_leaked", 0))),
                "naive_pii_total":  int(float(row.get("naive_pii_total",  0))),
                "naive_pii_rate":   float(row.get("naive_pii_rate", 0)),
                "naive_latency_s":  float(row.get("naive_latency_s", 0)),
            }
    print(f"📂 {len(naive_by_id)} résultats NaiveRAG chargés depuis {path.name}")
    return naive_by_id


# ── Métriques PII (ground-truth) ─────────────────────────────────────────────

def measure_pii_leakage_gt(response: str, chunks: list[dict]) -> tuple[int, int]:
    """
    Mesure la fuite PII ground-truth sur les chunks ORIGINAUX (raw_chunks).
    Retourne (pii_leaked, pii_total).
    """
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


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> list[dict]:
    if OPF_CHECKPOINT.exists():
        with open(OPF_CHECKPOINT, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint trouvé : {len(data)} queries déjà traitées — reprise automatique")
        return data
    return []


def save_checkpoint(results: list[dict]):
    with open(OPF_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def delete_checkpoint():
    OPF_CHECKPOINT.unlink(missing_ok=True)


# ── Runner OPF uniquement ─────────────────────────────────────────────────────

def run_opf_only(
    queries:      list[dict],
    naive_by_id:  dict[str, dict],
    opf_rag:      OPFNaiveRAG,
) -> list[dict]:
    """
    Lance OPFNaiveRAG sur chaque query.
    Les résultats NaiveRAG viennent directement du CSV existant.
    """
    results  = load_checkpoint()
    done_ids = {r["query_id"] for r in results}
    pending  = [q for q in queries if q["query_id"] not in done_ids]

    if not pending:
        print("  Toutes les queries sont déjà traitées (checkpoint complet).")
        return results

    print(f"  {len(pending)} queries restantes sur {len(queries)} total\n")

    done_total = len(results)   # queries déjà traitées avant cette session

    for i, q in enumerate(tqdm(pending, desc="OPFNaiveRAG"), start=1):
        query_text = str(q.get("query", ""))
        query_id   = q.get("global_id", q["query_id"])
        query_type = q["query_type"]

        # ── Log question dans le terminal ─────────────────────────────────────
        global_idx = done_total + i
        print(f"\n{'─'*70}")
        print(f"  [{global_idx}/{len(queries)}]  [{query_type}]  {query_id}")
        print(f"  ❓ QUESTION : {query_text}")
        print(f"{'─'*70}")

        # ── Résultats NaiveRAG depuis le CSV ──────────────────────────────────
        naive = naive_by_id.get(query_id, {})

        row = {
            "query_id":   query_id,
            "query_type": query_type,
            "query":      query_text[:300],
            # Résultats NaiveRAG (déjà calculés, lus depuis CSV)
            **naive,
        }

        # ── OPFNaiveRAG ───────────────────────────────────────────────────────
        # Pipeline : LLM génère sur chunks bruts → OPF redacte la RÉPONSE
        t0 = time.time()
        try:
            opf_out            = opf_rag.run(query_text, top_k=TOP_K)
            opf_resp           = opf_out.get("response", "")      # réponse finale (redactée)
            opf_raw_response   = opf_out.get("raw_response", "")  # réponse brute du LLM
            opf_raw_chunks     = opf_out.get("raw_chunks", [])
            opf_entities_total = opf_out.get("opf_entities_total", 0)  # entités dans la réponse
        except Exception as exc:
            opf_resp           = f"ERROR: {exc}"
            opf_raw_response   = ""
            opf_raw_chunks     = []
            opf_entities_total = 0

        opf_latency                   = round(time.time() - t0, 3)
        # Mesure PII sur la réponse APRÈS redaction OPF (ce qui sort réellement)
        opf_pii_leaked, opf_pii_total = measure_pii_leakage_gt(opf_resp, opf_raw_chunks)
        opf_pii_rate                  = (
            round(opf_pii_leaked / opf_pii_total, 4) if opf_pii_total > 0 else 0.0
        )

        # ── Log réponse dans le terminal ──────────────────────────────────────
        print(f"  🤖 LLM (brut)  : {opf_raw_response[:400]}")
        print(f"  🔒 OPF (filtré): {opf_resp[:400]}")
        print(f"  📊 PII leakée  : {opf_pii_leaked}/{opf_pii_total}"
              f"  |  Entités OPF : {opf_entities_total}"
              f"  |  ⏱ {opf_latency:.1f}s")

        row.update({
            "opf_response":       opf_resp,           # réponse finale (sans PII)
            "opf_raw_response":   opf_raw_response,   # réponse brute LLM (avant OPF)
            "opf_pii_leaked":     opf_pii_leaked,
            "opf_pii_total":      opf_pii_total,
            "opf_pii_rate":       opf_pii_rate,
            "opf_latency_s":      opf_latency,
            "opf_entities_total": opf_entities_total, # entités redactées dans la réponse
        })

        results.append(row)
        save_checkpoint(results)

    delete_checkpoint()
    return results


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(results: list[dict], llm_name: str):
    mlflow_uri = f"file:///{MLFLOW_DIR.replace(chr(92), '/')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"NaiveRAG_vs_OPF_{llm_name}"
    with mlflow.start_run(run_name=run_name):
        total = len(results)

        mlflow.log_param("llm",         llm_name)
        mlflow.log_param("n_queries",   total)
        mlflow.log_param("dataset",     "ildpil/text-anonymization-benchmark")
        mlflow.log_param("split",       "test")
        mlflow.log_param("opf_model",   "openai/privacy-filter")
        mlflow.log_param("condition_a", "NaiveRAG (résultats depuis benchmark_results.csv)")
        mlflow.log_param("condition_b", "OPFNaiveRAG (LLM génère sur contexte complet, OPF redacte la réponse)")

        def si(v):
            try: return int(float(v))
            except: return 0

        def sf(v):
            try: return float(v)
            except: return 0.0

        # ── Condition A : NaiveRAG (depuis CSV) ───────────────────────────────
        naive_leaked = sum(si(r.get("naive_pii_leaked", 0)) for r in results)
        naive_total  = sum(si(r.get("naive_pii_total",  0)) for r in results)
        naive_rate   = naive_leaked / naive_total if naive_total > 0 else 0.0
        naive_lat    = sum(sf(r.get("naive_latency_s",  0)) for r in results) / total

        mlflow.log_metric("naive_pii_leaked_total",   naive_leaked)
        mlflow.log_metric("naive_pii_total",          naive_total)
        mlflow.log_metric("naive_pii_leakage_rate",   round(naive_rate, 4))
        mlflow.log_metric("naive_latency_mean_s",     round(naive_lat,  3))

        # ── Condition B : OPFNaiveRAG ─────────────────────────────────────────
        opf_leaked   = sum(si(r.get("opf_pii_leaked",      0)) for r in results)
        opf_total    = sum(si(r.get("opf_pii_total",        0)) for r in results)
        opf_rate     = opf_leaked / opf_total if opf_total > 0 else 0.0
        opf_lat      = sum(sf(r.get("opf_latency_s",        0)) for r in results) / total
        opf_ent_mean = sum(si(r.get("opf_entities_total",   0)) for r in results) / total

        mlflow.log_metric("opf_pii_leaked_total",              opf_leaked)
        mlflow.log_metric("opf_pii_total",                     opf_total)
        mlflow.log_metric("opf_pii_leakage_rate",              round(opf_rate,     4))
        mlflow.log_metric("opf_latency_mean_s",                round(opf_lat,      3))
        mlflow.log_metric("opf_entities_in_response_mean",     round(opf_ent_mean, 4))

        # ── Réduction PII ─────────────────────────────────────────────────────
        reduction = (naive_rate - opf_rate) / naive_rate if naive_rate > 0 else 0.0
        mlflow.log_metric("pii_reduction_opf_vs_naive", round(reduction, 4))

        # ── Métriques par type de query ───────────────────────────────────────
        for qtype in sorted(set(r["query_type"] for r in results)):
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            if n == 0:
                continue
            s_naive_leaked = sum(si(r.get("naive_pii_leaked", 0)) for r in subset)
            s_naive_total  = sum(si(r.get("naive_pii_total",  0)) for r in subset)
            s_opf_leaked   = sum(si(r.get("opf_pii_leaked",   0)) for r in subset)
            s_opf_total    = sum(si(r.get("opf_pii_total",    0)) for r in subset)
            s_opf_ents     = sum(si(r.get("opf_entities_total", 0)) for r in subset) / n

            mlflow.log_metric(f"{qtype}_naive_pii_rate",
                round(s_naive_leaked / s_naive_total, 4) if s_naive_total > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_opf_pii_rate",
                round(s_opf_leaked   / s_opf_total,   4) if s_opf_total   > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_opf_entities_in_response_mean", round(s_opf_ents, 4))
            mlflow.log_metric(f"{qtype}_n_queries", n)

        # ── Table interactive MLflow ──────────────────────────────────────────
        table_data = {
            "query_id":            [r.get("query_id",   "")               for r in results],
            "query_type":          [r.get("query_type", "")               for r in results],
            "query":               [r.get("query",      "")[:300]         for r in results],
            "naive_response":      [r.get("naive_response", "")[:600]     for r in results],
            "naive_pii_leaked":    [si(r.get("naive_pii_leaked", 0))      for r in results],
            "naive_pii_total":     [si(r.get("naive_pii_total",  0))      for r in results],
            "naive_pii_rate":      [sf(r.get("naive_pii_rate",   0))      for r in results],
            "opf_response":        [r.get("opf_response", "")[:600]       for r in results],
            "opf_raw_response":    [r.get("opf_raw_response", "")[:600]   for r in results],
            "opf_pii_leaked":      [si(r.get("opf_pii_leaked",   0))      for r in results],
            "opf_pii_total":       [si(r.get("opf_pii_total",    0))      for r in results],
            "opf_pii_rate":        [sf(r.get("opf_pii_rate",     0))      for r in results],
            "opf_latency_s":       [sf(r.get("opf_latency_s",    0))      for r in results],
            "opf_entities_total":  [si(r.get("opf_entities_total", 0))    for r in results],
        }
        mlflow.log_table(table_data, artifact_file="results/naiveRAG_vs_OPF_table.json")
        print("  Table interactive : ✓")

        # ── CSV en artifact ───────────────────────────────────────────────────
        FIELDNAMES = [
            "query_id", "query_type", "query",
            "naive_response", "naive_pii_leaked", "naive_pii_total",
            "naive_pii_rate", "naive_latency_s",
            "opf_response", "opf_raw_response",
            "opf_pii_leaked", "opf_pii_total",
            "opf_pii_rate", "opf_latency_s",
            "opf_entities_total",
        ]
        with open(OPF_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        mlflow.log_artifact(str(OPF_RESULTS_CSV), artifact_path="results")
        print(f"  CSV : {OPF_RESULTS_CSV.name} ✓")

    print(f"\n  Expérience MLflow : {MLFLOW_EXPERIMENT}")
    print(f"\nPour visualiser : mlflow ui --backend-store-uri benchmark_naive_vs_cpb/mlruns")
    print(f"Puis ouvre      : http://127.0.0.1:5000")


# ── Résumé console ────────────────────────────────────────────────────────────

def print_summary(results: list[dict], llm_name: str):
    total = len(results)
    if total == 0:
        return

    def si(v):
        try: return int(float(v))
        except: return 0

    def sf(v):
        try: return float(v)
        except: return 0.0

    naive_leaked = sum(si(r.get("naive_pii_leaked", 0)) for r in results)
    naive_total  = sum(si(r.get("naive_pii_total",  0)) for r in results)
    naive_rate   = naive_leaked / naive_total if naive_total > 0 else 0.0
    naive_lat    = sum(sf(r.get("naive_latency_s",  0)) for r in results) / total

    opf_leaked   = sum(si(r.get("opf_pii_leaked",      0)) for r in results)
    opf_total    = sum(si(r.get("opf_pii_total",        0)) for r in results)
    opf_rate     = opf_leaked / opf_total if opf_total > 0 else 0.0
    opf_lat      = sum(sf(r.get("opf_latency_s",        0)) for r in results) / total
    opf_ents     = sum(si(r.get("opf_entities_total",   0)) for r in results) / total
    reduction    = (naive_rate - opf_rate) / naive_rate * 100 if naive_rate > 0 else 0.0

    print(f"\n{'='*68}")
    print(f"  RÉSULTATS — {total} queries — LLM : {llm_name}")
    print(f"  Pipeline : LLM génère sur contexte complet → OPF redacte la réponse")
    print(f"{'='*68}")
    print(f"  {'Métrique':<38} {'NaiveRAG (CSV)':>14}  {'OPF':>10}")
    print(f"  {'-'*65}")
    print(f"  {'PII leakage rate':<38} {naive_rate:>14.1%}  {opf_rate:>10.1%}")
    print(f"  {'PII leaked / total':<38} {naive_leaked}/{naive_total}  {opf_leaked}/{opf_total}")
    print(f"  {'Latence totale moy. (s)':<38} {naive_lat:>14.3f}  {opf_lat:>10.3f}")
    print(f"  {'Entités OPF dans réponse / query':<38} {'—':>14}  {opf_ents:>10.1f}")
    print(f"  {'-'*65}")
    print(f"  Réduction PII (OPF vs NaiveRAG) : {reduction:.1f}%")
    print(f"{'='*68}")

    print(f"\n  Détail par type de query :")
    print(f"  {'Type':<12} {'N':>4}  {'Naive PII%':>11}  {'OPF PII%':>10}  {'OPF ents moy':>14}")
    print(f"  {'-'*58}")
    for qtype in sorted(set(r["query_type"] for r in results)):
        s   = [r for r in results if r["query_type"] == qtype]
        n   = len(s)
        nl  = sum(si(r.get("naive_pii_leaked",   0)) for r in s)
        nt  = sum(si(r.get("naive_pii_total",    0)) for r in s)
        ol  = sum(si(r.get("opf_pii_leaked",     0)) for r in s)
        ot  = sum(si(r.get("opf_pii_total",      0)) for r in s)
        oe  = sum(si(r.get("opf_entities_total", 0)) for r in s) / n
        nr  = nl / nt if nt > 0 else 0.0
        or_ = ol / ot if ot > 0 else 0.0
        print(f"  {qtype:<12} {n:>4}  {nr:>11.1%}  {or_:>10.1%}  {oe:>14.1f}")

    print(f"\n  Résultats complets : {OPF_RESULTS_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OPFNaiveRAG — NaiveRAG chargé depuis le CSV existant"
    )
    parser.add_argument(
        "--llm", default="llama",
        choices=["llama", "mistral", "gpt4o-mini", "claude-haiku"],
        help="LLM pour la génération OPF"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limite le nombre de queries (test rapide)"
    )
    args = parser.parse_args()

    # Vérification pré-requises
    if not QUERIES_FILE.exists():
        print(f"ERREUR : {QUERIES_FILE} introuvable.")
        print("Lancez d'abord : python benchmark_naive_vs_cpb/01_generate_queries.py")
        sys.exit(1)

    if not RESULTS_CSV.exists():
        print(f"ERREUR : {RESULTS_CSV} introuvable.")
        print("Lancez d'abord : python benchmark_naive_vs_cpb/02_run_benchmark.py")
        sys.exit(1)

    # Chargement des queries
    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries = json.load(f)

    if args.limit:
        queries = queries[: args.limit]
        print(f"⚠️  Mode test : {args.limit} queries seulement")

    print(f"  {len(queries)} questions chargées")

    # Chargement des résultats NaiveRAG depuis le CSV existant (pas de relance !)
    naive_by_id = load_naive_results(RESULTS_CSV)

    # Vérification : est-ce que toutes les queries ont un résultat NaiveRAG ?
    missing = [q["query_id"] for q in queries if q["query_id"] not in naive_by_id]
    if missing:
        print(f"⚠️  {len(missing)} queries sans résultat NaiveRAG dans le CSV : {missing[:5]}...")

    # Initialisation ChromaDB
    print(f"\nInitialisation ChromaDB ({CHROMA_DIR})...")
    store = BenchmarkStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    if store.count() == 0:
        print("\nERREUR : collection ChromaDB vide.")
        print("Lancez d'abord : python test_contre_mesure_ildpiltest/01_index.py")
        sys.exit(1)

    # Initialisation LLM + OPF
    print(f"\nInitialisation LLM : {args.llm}...")
    llm       = build_llm(args.llm)
    naive_rag = NaiveRAG(store=store, llm=llm)
    opf_rag   = OPFNaiveRAG(naive_rag=naive_rag, architecture_name="opf_naive_rag")

    print(f"\nDémarrage du benchmark ({len(queries)} queries)...")
    print("  NaiveRAG    : résultats chargés depuis benchmark_results.csv ✓")
    print("  OPFNaiveRAG : lancement en cours...\n")

    results = run_opf_only(queries, naive_by_id, opf_rag)

    print(f"\nLogging dans MLflow ({MLFLOW_DIR})...")
    log_to_mlflow(results, args.llm)

    print_summary(results, args.llm)


if __name__ == "__main__":
    main()
