"""
run_ablation_cumulative_reverse.py — Reverse-cumulative ("knock-out from B7
inward") ablation for CPB v4 (Llama stack).

Same 300 Zhang et al. attack queries and corpus as run_ablation_cumulative.py,
but the layers are removed in REVERSE pipeline order, starting from the
output-side guard B7 and working back toward B0:
    cum_b7                 : B7
    cum_b7_b6              : B7 + B6
    cum_b7_b6_b3b4         : B7 + B6 + B3B4
    cum_b7_b6_b3b4_b1b2    : B7 + B6 + B3B4 + B1B2
    cum_all                : B7 + B6 + B3B4 + B1B2 + B0   (== naive RAG, all off)
B1+B2 and B3+B4 are each a single layer. Because B0 is disabled only at the very
last step, the B0->B6 coupling never fires early, so B6 gets its own clean
cumulative step here (switched off explicitly at cum_b7_b6, B0 still on).
B5 (LLM.generate) is never ablated.

This script does NOT modify countermeasure_v4/cpb_ablation.py: it imports only
the shared AblationConfig dataclass and defines its own variant list below, so
the leave-one-out and forward-cumulative studies are untouched.

The full-pipeline baseline is NOT rerun here: it is reused as-is from
data/zhang_eval/results.json.

OUTPUT LOCATION: everything is written next to THIS script, under
evaluation_zhang_ablation/cumulative_results_reverse/<variant_name>/ — nothing
is written under data/. The shared inputs (doc_index, attack_queries, baseline
results.json) are still READ from data/zhang_eval/ (reused as-is).

IMPORT ORDER (same native-Windows DLL trap as the other ablation scripts):
torch (via spacy) and mlflow loaded in the wrong order in one process trigger a
DLL-load access violation. Safe, verified order:
  1. CPB generation (spacy/torch, then chromadb) for ALL variants first —
     mlflow must not be imported yet.
  2. Only once every variant has finished generating + scoring, import mlflow
     and log everything.
Hence Phase 1 (generation+scoring) and Phase 2 (mlflow) are two separate passes.

Usage:
  python run_ablation_cumulative_reverse.py [--skip-generation] [--variants cum_b7 ...]
    --skip-generation  : reuse cached responses per variant if present
    --variants         : run only a subset of variants (default: all 5)
"""
from __future__ import annotations  # keep type hints as strings -> no eager import of annotated types

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")        # ChromaDB's Rust core
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_zhang"))

from config import LLAMA_MODEL, MLFLOW_TRACKING_URI  # noqa: E402 — lightweight, no heavy deps

SHARED_DATA_DIR = Path(__file__).parent.parent / "data" / "zhang_eval"           # doc_index, attack_queries, reference_responses (read-only, shared)
BASELINE_RESULTS_PATH = SHARED_DATA_DIR / "results.json"                        # full-pipeline baseline, reused as-is
OUT_ROOT = Path(__file__).parent / "cumulative_results_reverse"                  # all reverse-cumulative outputs live next to this script
CHROMA_ZHANG_DIR = Path(__file__).parent.parent / "data" / "chroma_zhang"
EXPERIMENT_NAME = "zhang_evaluation_ablation_cumulative_reverse"

METRIC_ORDER = ["LO_F1", "AE", "PI", "CR", "SS", "AR"]


# ── Reverse-cumulative variants — defined HERE so cpb_ablation.py is untouched ─
# Uses only the shared AblationConfig dataclass. Each variant disables one MORE
# layer than the previous, in reverse pipeline order (B7 -> B6 -> B3B4 -> B1B2
# -> B0). B1+B2 and B3+B4 are each a single layer; B5 is never ablated.

def build_reverse_cumulative_variants():
    from countermeasure_v4.cpb_ablation import AblationConfig  # spacy/torch load here, before chromadb/mlflow
    return [
        AblationConfig(
            name="cum_b7",
            b7_response_guard=False,
        ),
        AblationConfig(
            name="cum_b7_b6",
            b7_response_guard=False,
            b6_sad_detector=False,
        ),
        AblationConfig(
            name="cum_b7_b6_b3b4",
            b7_response_guard=False,
            b6_sad_detector=False,
            b3_pii_analyzer=False, b4_pii_anonymizer=False,
        ),
        AblationConfig(
            name="cum_b7_b6_b3b4_b1b2",
            b7_response_guard=False,
            b6_sad_detector=False,
            b3_pii_analyzer=False, b4_pii_anonymizer=False,
            b1_query_risk=False, b2_budget_gate=False,
        ),
        AblationConfig(
            name="cum_all",
            b7_response_guard=False,
            b6_sad_detector=False,
            b3_pii_analyzer=False, b4_pii_anonymizer=False,
            b1_query_risk=False, b2_budget_gate=False,
            b0_bootstrap=False,                                                    # -> b6 already off
        ),
    ]


# ── ChromaStore wrapper (identical to evaluation_zhang/run_evaluation.py) ─────

class ZhangChromaStore:
    def __init__(self):
        import chromadb
        from chromadb.config import Settings

        from embeddings.embedder import Embedder

        self._embedder = Embedder()
        client = chromadb.PersistentClient(
            path=str(CHROMA_ZHANG_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection("zhang_eval_corpus")
        print(f"ZhangChromaStore ready: {self.collection.count()} chunks")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        query_emb = self._embedder.embed_single(query_text).tolist()
        n_results = min(top_k * 3, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        seen_doc_ids: set[str] = set()

        for j in range(len(results["ids"][0])):
            meta = results["metadatas"][0][j]
            doc_id = meta.get("source_doc_id", results["ids"][0][j])

            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            chunks.append({
                "chunk_id":         results["ids"][0][j],
                "text":             results["documents"][0][j],
                "similarity_score": 1.0 - results["distances"][0][j],
                "doc_id":           doc_id,
                "n_pii":            0,
                "pii_entities":     [],
            })

            if len(chunks) >= top_k:
                break

        return chunks

    def get(self, limit: int = 50, include: list | None = None) -> dict:
        return self.collection.get(limit=limit, include=include or ["documents"])

    def count(self) -> int:
        return self.collection.count()


# ── CPB v4 inference, one reverse-cumulative variant ──────────────────────────

def run_cpb_v4(attacks: list[dict], ablation) -> tuple[list[str], list[list[str]]]:
    from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG

    store = ZhangChromaStore()
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV4(naive_rag=naive_rag, ablation=ablation)

    responses: list[str] = []
    contexts_per_query: list[list[str]] = []

    for i, attack in enumerate(attacks):
        print(f"  CPB v4 [{ablation.name}] [{i + 1}/{len(attacks)}] {attack['doc_id']}...", end="\r")
        result = cpb.run(attack["query"])
        responses.append(result["response"])
        chunk_texts = [c.get("text", "") for c in result.get("chunks", [])]
        contexts_per_query.append(chunk_texts)

    print()
    return responses, contexts_per_query


def load_or_run_cpb(
    attacks: list[dict], ablation, variant_dir: Path, skip_generation: bool,
) -> tuple[list[str], list[list[str]]]:
    responses_path = variant_dir / "responses.json"
    contexts_path = variant_dir / "contexts.json"

    if skip_generation and responses_path.exists() and contexts_path.exists():
        print(f"  Loading cached responses for {ablation.name}...")
        with open(responses_path, encoding="utf-8") as f:
            responses = json.load(f)
        with open(contexts_path, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"  {len(responses)} responses loaded from cache.")
        return responses, contexts_per_query

    print(f"  Running CPB v4 [{ablation.name}] (llama3.1:8b)...")
    responses, contexts_per_query = run_cpb_v4(attacks, ablation)

    variant_dir.mkdir(parents=True, exist_ok=True)
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    with open(contexts_path, "w", encoding="utf-8") as f:
        json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)

    print(f"  Responses saved -> {responses_path}")
    return responses, contexts_per_query


# ── Phase 1: generation + metrics for one variant — NO mlflow here ───────────

def generate_and_score(
    ablation,
    doc_index: dict,
    attacks: list[dict],
    skip_generation: bool,
) -> dict:
    variant_dir = OUT_ROOT / ablation.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    responses, contexts_per_query = load_or_run_cpb(attacks, ablation, variant_dir, skip_generation)

    print(f"  [{ablation.name}] [LO] ROUGE-L...")
    from metric_lo import aggregate_lo, compute_lo
    lo_results = [
        compute_lo(resp, doc_index.get(attack["doc_id"], {}).get("text", ""))
        for resp, attack in zip(responses, attacks)
    ]
    lo_agg = aggregate_lo(lo_results)

    print(f"  [{ablation.name}] [AE] GPT-4o judge...")
    from metric_ae import aggregate_ae, compute_ae_batch
    ae_cache_path = variant_dir / "ae_results.json"
    if ae_cache_path.exists():
        with open(ae_cache_path, encoding="utf-8") as f:
            ae_results = json.load(f)
    else:
        ae_results = compute_ae_batch(responses, attacks, verbose=True)
        with open(ae_cache_path, "w", encoding="utf-8") as f:
            json.dump(ae_results, f, ensure_ascii=False, indent=2)
    ae_score = aggregate_ae(ae_results)

    print(f"  [{ablation.name}] [PI] Personal Identification...")
    from metric_pi import PIMetric
    pi_cache_path = variant_dir / "pi_scores.json"
    pi_metric = PIMetric()
    if pi_cache_path.exists():
        with open(pi_cache_path, encoding="utf-8") as f:
            pi_scores = json.load(f)
    else:
        # build_claims_db reuses the claims DB cache already built for the
        # baseline run (it depends only on doc_index, not on the variant).
        pi_metric.build_claims_db(doc_index)
        pi_scores = pi_metric.compute_pi_batch(responses, attacks, verbose=True)
        with open(pi_cache_path, "w", encoding="utf-8") as f:
            json.dump(pi_scores, f, ensure_ascii=False, indent=2)
    pi_score = PIMetric.aggregate_pi(pi_scores)

    print(f"  [{ablation.name}] Utility metrics (RAGAS + GPT-4o)...")
    from metric_utility import compute_utility, generate_reference_responses
    utility_cache_path = variant_dir / "utility_scores.json"
    # generate_reference_responses reads/writes evaluation_zhang's shared
    # cache (data/zhang_eval/reference_responses.json) -- reused as-is,
    # independent of the ablation variant.
    references = generate_reference_responses(attacks, doc_index)
    if utility_cache_path.exists():
        with open(utility_cache_path, encoding="utf-8") as f:
            utility = json.load(f)
    else:
        utility = compute_utility(attacks, responses, contexts_per_query, references)
        with open(utility_cache_path, "w", encoding="utf-8") as f:
            json.dump(utility, f, ensure_ascii=False, indent=2)

    print(f"  [{ablation.name}] LO_F1={lo_agg['f1']:.4f}  AE={ae_score:.4f}  PI={pi_score:.4f}  "
          f"CR={utility['CR']:.4f}  SS={utility['SS']:.4f}  AR={utility['AR']:.4f}")

    # CSV export -- one row per query
    csv_path = variant_dir / "results_per_query.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "doc_id", "query", "response",
            "LO_precision", "LO_recall", "LO_f1",
            "AE", "PI",
        ])
        writer.writeheader()
        for i, (attack, resp) in enumerate(zip(attacks, responses)):
            lo = lo_results[i] if i < len(lo_results) else {}
            writer.writerow({
                "index":        i,
                "doc_id":       attack.get("doc_id", ""),
                "query":        attack.get("query", ""),
                "response":     resp,
                "LO_precision": round(lo.get("precision", 0.0), 4),
                "LO_recall":    round(lo.get("recall",    0.0), 4),
                "LO_f1":        round(lo.get("f1",        0.0), 4),
                "AE":           ae_results[i]["score"] if i < len(ae_results) else "",
                "PI":           round(pi_scores[i],   4) if i < len(pi_scores)   else "",
            })

    metrics = {
        "LO_F1": lo_agg["f1"],
        "AE":    ae_score,
        "PI":    pi_score,
        "CR":    utility["CR"],
        "SS":    utility["SS"],
        "AR":    utility["AR"],
    }
    variant_results = {
        "system":            "cpb_v4_cumulative_ablation_reverse",
        "ablation_variant":  ablation.name,
        "ablation_flags":    ablation.__dict__,
        "llm":               LLAMA_MODEL,
        "n_instances":       len(attacks),
        "metrics":           metrics,
        "per_instance": {
            "LO": lo_results,
            "AE": ae_results,
            "PI": pi_scores,
        },
    }
    results_path = variant_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(variant_results, f, ensure_ascii=False, indent=2)

    return variant_results


# ── Phase 2: mlflow logging for every variant — called once, at the very end ──

def log_all_to_mlflow(attacks: list[dict], variant_results: list[dict]) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for vr in variant_results:
        flags = vr["ablation_flags"]
        variant_dir = OUT_ROOT / vr["ablation_variant"]
        with mlflow.start_run(run_name=f"cpb_v4_cumulative_ablation_reverse_{vr['ablation_variant']}"):
            mlflow.log_param("system", "cpb_v4")
            mlflow.log_param("ablation_variant", vr["ablation_variant"])
            for flag_name, flag_value in flags.items():
                if flag_name != "name":
                    mlflow.log_param(flag_name, flag_value)
            mlflow.log_param("llm_generation", LLAMA_MODEL)
            mlflow.log_param("llm_evaluation", "gpt-4o")
            mlflow.log_param("dataset", "umarbutler/open-australian-legal-corpus")
            mlflow.log_param("n_queries", len(attacks))
            mlflow.log_param("n_responses", vr["n_instances"])
            for name, value in vr["metrics"].items():
                mlflow.log_metric(name, value)
            mlflow.log_artifact(str(variant_dir / "results.json"))
            mlflow.log_artifact(str(variant_dir / "results_per_query.csv"))


# ── Summary comparison vs the full-pipeline baseline ──────────────────────────

def print_and_save_summary(baseline_metrics: dict, variant_results: list[dict]) -> None:
    summary_path = OUT_ROOT / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + METRIC_ORDER + [f"d_{m}" for m in METRIC_ORDER])
        writer.writerow(["full_pipeline"] + [round(baseline_metrics[m], 4) for m in METRIC_ORDER] + [0.0] * len(METRIC_ORDER))
        for vr in variant_results:
            m = vr["metrics"]
            deltas = [round(m[k] - baseline_metrics[k], 4) for k in METRIC_ORDER]
            writer.writerow([vr["ablation_variant"]] + [round(m[k], 4) for k in METRIC_ORDER] + deltas)

    print("\n" + "=" * 100)
    print("  REVERSE-CUMULATIVE ABLATION SUMMARY -- CPB v4, Zhang legal corpus (300 queries)")
    print("=" * 100)
    header = f"  {'Variant':<24}" + "".join(f"{m:>10}" for m in METRIC_ORDER)
    print(header)
    print("-" * 100)
    print(f"  {'full_pipeline':<24}" + "".join(f"{baseline_metrics[m]:>10.4f}" for m in METRIC_ORDER))
    for vr in variant_results:
        m = vr["metrics"]
        print(f"  {vr['ablation_variant']:<24}" + "".join(f"{m[k]:>10.4f}" for k in METRIC_ORDER))
    print("=" * 100)
    print(f"  Saved -> {summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(skip_generation: bool = False, variant_names: list[str] | None = None):
    all_variants = build_reverse_cumulative_variants()  # spacy/torch load here, before chromadb/mlflow

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=== CPB v4 Reverse-Cumulative Ablation Study -- Zhang legal corpus ===\n")

    print("1. Loading shared data (doc_index, attack_queries, baseline)...")
    with open(SHARED_DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        doc_index = json.load(f)
    attacks = json.loads((SHARED_DATA_DIR / "attack_queries.json").read_text(encoding="utf-8"))
    with open(BASELINE_RESULTS_PATH, encoding="utf-8") as f:
        baseline_metrics = json.load(f)["metrics"]
    print(f"   {len(doc_index)} documents, {len(attacks)} attack queries")
    print(f"   Baseline (full pipeline) loaded from {BASELINE_RESULTS_PATH}\n")

    variants = all_variants if not variant_names else [v for v in all_variants if v.name in variant_names]

    # ── Phase 1: all ChromaDB/torch work, all variants. mlflow stays unimported. ──
    variant_results = []
    for i, ablation in enumerate(variants):
        print(f"2.{i + 1} Variant {ablation.name} ({i + 1}/{len(variants)})...")
        variant_results.append(generate_and_score(ablation, doc_index, attacks, skip_generation))
        print()

    # ── Phase 2: mlflow logging only, now that ChromaDB/torch are done for good. ──
    print("3. MLflow logging...")
    log_all_to_mlflow(attacks, variant_results)

    print_and_save_summary(baseline_metrics, variant_results)
    print(f"\nDone. Per-variant results under {OUT_ROOT}\\<variant>\\results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPB v4 reverse-cumulative ablation study (Zhang legal corpus)")
    parser.add_argument("--skip-generation", action="store_true",
                         help="Reuse cached responses per variant if present")
    parser.add_argument("--variants", nargs="+", default=None,
                         help="Run only these variants (default: all 5). "
                              "Choices: cum_b7 cum_b7_b6 cum_b7_b6_b3b4 cum_b7_b6_b3b4_b1b2 cum_all")
    args = parser.parse_args()
    main(skip_generation=args.skip_generation, variant_names=args.variants)
