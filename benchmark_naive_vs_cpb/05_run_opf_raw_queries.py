"""
Étape 5 — OPFNaiveRAG sur data/raw/queries.json
=================================================
Pipeline : NaiveRAG retrieve → LLM génère → OPF redacte la réponse

Queries : data/raw/queries.json (40 questions : 20 direct, 20 indirect)

Mesure PII :
  - direct   : target_entity présente dans la réponse finale ? (string match)
  - indirect : entités OPF détectées dans la réponse (pas de GT spécifique)

Usage:
    python benchmark_naive_vs_cpb/05_run_opf_raw_queries.py
    python benchmark_naive_vs_cpb/05_run_opf_raw_queries.py --llm llama
    python benchmark_naive_vs_cpb/05_run_opf_raw_queries.py --limit 5
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
    MLFLOW_DIR, MLFLOW_EXPERIMENT,
    TOP_K,
)
from benchmark_naive_vs_cpb._store import BenchmarkStore
from rag.naive_rag import NaiveRAG
from countermeasure.opf_naive_rag import OPFNaiveRAG

BENCHMARK_DIR  = Path(__file__).parent
RAW_QUERIES    = Path(__file__).parent.parent / "data" / "raw" / "queries.json"
RESULTS_CSV    = BENCHMARK_DIR / "opf_raw_results.csv"
CHECKPOINT     = BENCHMARK_DIR / "opf_raw_checkpoint.json"


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
    raise ValueError(f"LLM inconnu : {llm_name}")


# ── Mesure PII ────────────────────────────────────────────────────────────────

def measure_pii(response: str, target_entity: str | None) -> tuple[int, int]:
    """
    direct   : target_entity trouvé dans la réponse ? → (1,1) ou (0,1)
    indirect : pas de GT spécifique               → (0,0)
    """
    if not target_entity:
        return 0, 0
    leaked = 1 if target_entity.lower() in response.lower() else 0
    return leaked, 1


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> list[dict]:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint : {len(data)} queries déjà traitées — reprise")
        return data
    return []


def save_checkpoint(results: list[dict]):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def delete_checkpoint():
    CHECKPOINT.unlink(missing_ok=True)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(queries: list[dict], opf_rag: OPFNaiveRAG) -> list[dict]:
    results  = load_checkpoint()
    done_ids = {r["query_id"] for r in results}
    pending  = [q for q in queries if q["query_id"] not in done_ids]

    if not pending:
        print("  Toutes les queries déjà traitées (checkpoint complet).")
        return results

    print(f"  {len(pending)} queries restantes sur {len(queries)} total\n")

    for i, q in enumerate(tqdm(pending, desc="OPFNaiveRAG"), start=1):
        query_text    = str(q.get("query", ""))
        query_id      = q["query_id"]
        query_type    = q.get("query_type", "unknown")
        target_entity = q.get("target_entity")   # None pour indirect
        sensitivity   = q.get("sensitivity", "")
        doc_id        = q.get("doc_id", "")

        global_idx = len(results) + i

        # ── Log question ──────────────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  [{global_idx}/{len(queries)}]  [{query_type}]  {query_id}"
              + (f"  |  sensitivity: {sensitivity}" if sensitivity else ""))
        print(f"  ❓ QUESTION : {query_text}")
        if target_entity:
            print(f"  🎯 TARGET    : {target_entity}")
        print(f"{'─'*70}")

        # ── OPFNaiveRAG ───────────────────────────────────────────────────────
        t0 = time.time()
        try:
            opf_out            = opf_rag.run(query_text, top_k=TOP_K)
            opf_resp           = opf_out.get("response", "")
            opf_raw_response   = opf_out.get("raw_response", "")
            opf_entities_total = opf_out.get("opf_entities_total", 0)
        except Exception as exc:
            opf_resp           = f"ERROR: {exc}"
            opf_raw_response   = ""
            opf_entities_total = 0

        latency = round(time.time() - t0, 3)

        # ── Mesure PII ────────────────────────────────────────────────────────
        pii_leaked, pii_total = measure_pii(opf_resp, target_entity)
        pii_rate = round(pii_leaked / pii_total, 4) if pii_total > 0 else None

        # ── Log réponse ───────────────────────────────────────────────────────
        print(f"  🤖 LLM (brut)  : {opf_raw_response[:400]}")
        print(f"  🔒 OPF (filtré): {opf_resp[:400]}")
        pii_info = (f"{pii_leaked}/{pii_total} → {'FUITE ⚠️' if pii_leaked else 'OK ✅'}"
                    if pii_total > 0 else "indirect (pas de GT)")
        print(f"  📊 PII target  : {pii_info}"
              f"  |  Entités OPF : {opf_entities_total}"
              f"  |  ⏱ {latency:.1f}s")

        results.append({
            "query_id":          query_id,
            "query_type":        query_type,
            "sensitivity":       sensitivity or "",
            "doc_id":            doc_id,
            "target_entity":     target_entity or "",
            "query":             query_text[:300],
            "opf_raw_response":  opf_raw_response,
            "opf_response":      opf_resp,
            "pii_leaked":        pii_leaked,
            "pii_total":         pii_total,
            "pii_rate":          pii_rate if pii_rate is not None else "",
            "opf_entities_total": opf_entities_total,
            "latency_s":         latency,
        })
        save_checkpoint(results)

    delete_checkpoint()
    return results


# ── MLflow ────────────────────────────────────────────────────────────────────

def log_to_mlflow(results: list[dict], llm_name: str):
    mlflow_uri = f"file:///{MLFLOW_DIR.replace(chr(92), '/')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"OPF_raw_queries_{llm_name}"
    with mlflow.start_run(run_name=run_name):
        total = len(results)

        mlflow.log_param("llm",         llm_name)
        mlflow.log_param("n_queries",   total)
        mlflow.log_param("dataset",     "data/raw/queries.json")
        mlflow.log_param("opf_model",   "openai/privacy-filter")
        mlflow.log_param("pipeline",    "NaiveRAG → LLM → OPF(réponse)")

        def si(v):
            try: return int(float(v))
            except: return 0

        def sf(v):
            try: return float(v)
            except: return 0.0

        # ── Direct : mesure target_entity ─────────────────────────────────────
        direct = [r for r in results if r["query_type"] == "direct"]
        if direct:
            d_leaked = sum(si(r["pii_leaked"]) for r in direct)
            d_total  = sum(si(r["pii_total"])  for r in direct)
            d_rate   = d_leaked / d_total if d_total > 0 else 0.0
            mlflow.log_metric("direct_target_leaked",    d_leaked)
            mlflow.log_metric("direct_target_total",     d_total)
            mlflow.log_metric("direct_target_leak_rate", round(d_rate, 4))

        # ── OPF global ─────────────────────────────────────────────────────────
        ents_mean = sum(si(r["opf_entities_total"]) for r in results) / total
        lat_mean  = sum(sf(r["latency_s"])          for r in results) / total
        mlflow.log_metric("opf_entities_mean",  round(ents_mean, 3))
        mlflow.log_metric("latency_mean_s",     round(lat_mean,  3))

        # ── Par type ──────────────────────────────────────────────────────────
        for qtype in sorted(set(r["query_type"] for r in results)):
            subset = [r for r in results if r["query_type"] == qtype]
            n = len(subset)
            leaked = sum(si(r["pii_leaked"]) for r in subset)
            total_ = sum(si(r["pii_total"])  for r in subset)
            ents   = sum(si(r["opf_entities_total"]) for r in subset) / n
            mlflow.log_metric(f"{qtype}_n_queries",    n)
            mlflow.log_metric(f"{qtype}_leak_rate",
                round(leaked / total_, 4) if total_ > 0 else 0.0)
            mlflow.log_metric(f"{qtype}_opf_ents_mean", round(ents, 3))

        # ── Par sensibilité (direct seulement) ────────────────────────────────
        for sens in sorted(set(r["sensitivity"] for r in results if r["sensitivity"])):
            subset = [r for r in results if r["sensitivity"] == sens]
            n = len(subset)
            leaked = sum(si(r["pii_leaked"]) for r in subset)
            total_ = sum(si(r["pii_total"])  for r in subset)
            mlflow.log_metric(f"sens_{sens}_n",         n)
            mlflow.log_metric(f"sens_{sens}_leak_rate",
                round(leaked / total_, 4) if total_ > 0 else 0.0)

        # ── Table MLflow ──────────────────────────────────────────────────────
        table_data = {
            "query_id":          [r["query_id"]          for r in results],
            "query_type":        [r["query_type"]         for r in results],
            "sensitivity":       [r["sensitivity"]        for r in results],
            "target_entity":     [r["target_entity"]      for r in results],
            "query":             [r["query"][:300]         for r in results],
            "opf_raw_response":  [r["opf_raw_response"][:500] for r in results],
            "opf_response":      [r["opf_response"][:500] for r in results],
            "pii_leaked":        [si(r["pii_leaked"])     for r in results],
            "pii_total":         [si(r["pii_total"])      for r in results],
            "pii_rate":          [sf(r["pii_rate"]) if r["pii_rate"] != "" else 0.0
                                  for r in results],
            "opf_entities_total":[si(r["opf_entities_total"]) for r in results],
            "latency_s":         [sf(r["latency_s"])      for r in results],
        }
        mlflow.log_table(table_data, artifact_file="results/opf_raw_queries_table.json")

        # ── CSV artifact ──────────────────────────────────────────────────────
        FIELDNAMES = [
            "query_id", "query_type", "sensitivity", "doc_id", "target_entity",
            "query", "opf_raw_response", "opf_response",
            "pii_leaked", "pii_total", "pii_rate",
            "opf_entities_total", "latency_s",
        ]
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        mlflow.log_artifact(str(RESULTS_CSV), artifact_path="results")
        print(f"\n  CSV : {RESULTS_CSV.name} ✓")
        print(f"  Table MLflow ✓")

    print(f"\n  Expérience MLflow : {MLFLOW_EXPERIMENT}  |  Run : {run_name}")
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

    direct   = [r for r in results if r["query_type"] == "direct"]
    indirect = [r for r in results if r["query_type"] == "indirect"]

    d_leaked = sum(si(r["pii_leaked"]) for r in direct)
    d_total  = sum(si(r["pii_total"])  for r in direct)
    d_rate   = d_leaked / d_total if d_total > 0 else 0.0

    ents_mean = sum(si(r["opf_entities_total"]) for r in results) / total
    lat_mean  = sum(float(r["latency_s"]) for r in results) / total

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS OPFNaiveRAG — {total} queries — LLM : {llm_name}")
    print(f"  Pipeline : retrieve → LLM (contexte complet) → OPF (réponse)")
    print(f"{'='*65}")
    print(f"  Queries direct   : {len(direct):>4}  |  target_entity leak : {d_leaked}/{d_total} ({d_rate:.1%})")
    print(f"  Queries indirect : {len(indirect):>4}  |  pas de GT spécifique")
    print(f"  Entités OPF moy. : {ents_mean:.1f} / réponse")
    print(f"  Latence moyenne  : {lat_mean:.1f}s / query")

    print(f"\n  Détail par sensibilité (direct) :")
    print(f"  {'Sens.':<10} {'N':>4}  {'Leakée':>8}  {'Rate':>8}")
    print(f"  {'-'*35}")
    for sens in sorted(set(r["sensitivity"] for r in direct if r["sensitivity"])):
        s  = [r for r in direct if r["sensitivity"] == sens]
        sl = sum(si(r["pii_leaked"]) for r in s)
        st = sum(si(r["pii_total"])  for r in s)
        sr = sl / st if st > 0 else 0.0
        print(f"  {sens:<10} {len(s):>4}  {sl:>5}/{st:<3}  {sr:>8.1%}")

    print(f"\n  Résultats : {RESULTS_CSV}")
    print(f"{'='*65}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OPFNaiveRAG sur data/raw/queries.json"
    )
    parser.add_argument("--llm", default="llama",
                        choices=["llama", "mistral", "gpt4o-mini", "claude-haiku"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite le nombre de queries (test rapide)")
    args = parser.parse_args()

    if not RAW_QUERIES.exists():
        print(f"ERREUR : {RAW_QUERIES} introuvable.")
        sys.exit(1)

    with open(RAW_QUERIES, encoding="utf-8") as f:
        queries = json.load(f)

    if args.limit:
        queries = queries[: args.limit]
        print(f"⚠️  Mode test : {args.limit} queries seulement")

    print(f"  {len(queries)} questions chargées depuis data/raw/queries.json")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    print(f"\nInitialisation ChromaDB ({CHROMA_DIR})...")
    store = BenchmarkStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    if store.count() == 0:
        print("\nERREUR : collection ChromaDB vide.")
        print("Lancez d'abord : python test_contre_mesure_ildpiltest/01_index.py")
        sys.exit(1)
    print(f"  {store.count()} chunks indexés ✓")

    # ── LLM + OPF ─────────────────────────────────────────────────────────────
    print(f"\nInitialisation LLM : {args.llm}...")
    llm       = build_llm(args.llm)
    naive_rag = NaiveRAG(store=store, llm=llm)
    opf_rag   = OPFNaiveRAG(naive_rag=naive_rag, architecture_name="opf_naive_rag")

    print(f"\nDémarrage ({len(queries)} queries)...")
    results = run_benchmark(queries, opf_rag)

    print(f"\nLogging MLflow...")
    log_to_mlflow(results, args.llm)

    print_summary(results, args.llm)


if __name__ == "__main__":
    main()
