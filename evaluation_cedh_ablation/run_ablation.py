"""
run_ablation.py — Leave-one-out ablation study for CPB v4 on the CEDH corpus
(ildpil/text-anonymization-benchmark, ECHR case law).

Reuses the already-indexed ChromaDB collection and the 1000 attack queries
generated under test_contre_mesure_ildpiltest/ (chroma_db/, queries.json) —
nothing is re-indexed or regenerated here. Unlike that benchmark's inline
metrics, responses are scored with the ORIGINAL project metrics:
  - metrics/pii_leakage.py    : ground-truth PII leakage rate (ratio of
                                 sensitive entities annotated in the
                                 retrieved chunks that leak into the
                                 response)
  - metrics/response_quality.py : exact match, ROUGE-L, BERTScore F1,
                                 answer relevancy, weighted quality score
These are the metrics used by the very first experiments/ scripts and
app.py — distinct from the LO/AE/PI/CR/SS/AR Zhang-style metrics used by
evaluation_zhang_ablation/ (which need a GPT-4o judge + reference
responses; nothing here calls an external API).

CPB v4 has never been run on this dataset before, so unlike
evaluation_zhang_ablation (which reuses an existing full-pipeline baseline
from data/zhang_eval/results.json), this script also runs the default
AblationConfig() ("full_pipeline") as one of the 6 variants — see
countermeasure_v4/cpb_ablation.py VARIANTS for the other 5 leave-one-out
configs.

Sample: a fixed, stratified 300-query subset of the 1000 generated queries
(same per-type ratio as the full set: 90 normal / 90 direct / 60 injection /
30 dgea / 30 mia), seeded for reproducibility and cached to
data/cedh_eval_ablation/sampled_queries.json so every variant scores the
exact same queries.

IMPORTANT — IMPORT ORDER: see evaluation_zhang_ablation/run_ablation.py for
why mlflow must only be imported after every torch-touching call has
finished for every variant. BERTScore also loads a torch model, so it is
computed in Phase 1 alongside CPB v4 generation (and batched once per
variant — see compute_response_quality's precomputed_bert_f1 param —
because bert_score.score() reloads roberta-large from scratch on every
single call with no caching, which would otherwise add over an hour of
pure reload overhead across 300 queries x 6 variants).

Usage:
  python run_ablation.py [--skip-generation] [--variants full_pipeline b0_off ...]
    [--n-queries 300] [--seed 42]
    --skip-generation : reuse cached responses/chunks per variant if present
    --variants         : run only a subset (default: all 6)
    --n-queries        : size of the stratified query sample (default: 300)
    --seed             : sampling seed (default: 42)
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
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LLAMA_MODEL, MLFLOW_TRACKING_URI, TOP_K  # noqa: E402 — lightweight, no heavy deps
from test_contre_mesure_ildpiltest.config import (  # noqa: E402
    CHROMA_DIR, COLLECTION_NAME, QUERIES_FILE as SHARED_QUERIES_FILE,
)

OUT_ROOT = Path(__file__).parent.parent / "data" / "cedh_eval_ablation"  # all ablation outputs live here
SAMPLED_QUERIES_PATH = OUT_ROOT / "sampled_queries.json"
EXPERIMENT_NAME = "cedh_evaluation_ablation"

METRIC_ORDER = ["PII", "QS", "AR", "RL", "BF1", "EM"]
METRIC_NAMES = {
    "PII": "PII leakage rate",
    "QS":  "Quality score",
    "AR":  "Answer relevancy",
    "RL":  "ROUGE-L",
    "BF1": "BERTScore F1",
    "EM":  "Exact match",
}

ENTITY_HINT_RE = re.compile(r"^(.*) \([A-Z_]+\)$")


def parse_target_entity(entity_hint: str) -> str | None:
    """Extracts the entity text from an "<text> (<TYPE>)" hint, e.g.
    "Mr Gunnar Beck (PERSON)" -> "Mr Gunnar Beck". Returns None for the
    generic fallback hints ("an individual or organization", ...)."""
    m = ENTITY_HINT_RE.match(entity_hint or "")
    return m.group(1).strip() if m else None


def get_query_text(q: dict) -> str:
    """95/1000 dgea-type entries in queries.json have a malformed "query"
    field -- gpt-4o-mini sometimes returned {"type": "...", "query": "..."}
    instead of a plain string inside the "questions" array (see
    02_generate_queries.py's build_prompt_dgea). Extract the inner text
    instead of stringifying the whole dict (which would pollute the attack
    query with literal dict-repr noise)."""
    text = q.get("query", "")
    if isinstance(text, dict):
        return text.get("query") or str(text)
    return text if isinstance(text, str) else str(text)


# ── Stratified query sample, cached once and reused by every variant ─────────

def load_sampled_queries(n_total: int, seed: int) -> list[dict]:
    if SAMPLED_QUERIES_PATH.exists():
        with open(SAMPLED_QUERIES_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == n_total:
            print(f"  Reusing cached sample -> {SAMPLED_QUERIES_PATH} ({n_total} queries)")
            return cached
        print(f"  Cached sample has {len(cached)} queries, requested {n_total} — resampling")

    with open(SHARED_QUERIES_FILE, encoding="utf-8") as f:
        all_queries = json.load(f)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in all_queries:
        by_type[q["query_type"]].append(q)

    ratio = n_total / len(all_queries)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for qtype in sorted(by_type):
        items = by_type[qtype][:]
        rng.shuffle(items)
        k = round(len(items) * ratio)
        sampled.extend(items[:k])
    rng.shuffle(sampled)
    sampled = sampled[:n_total]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SAMPLED_QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    counts = defaultdict(int)
    for q in sampled:
        counts[q["query_type"]] += 1
    print(f"  Sampled {len(sampled)} queries (seed={seed}): {dict(sorted(counts.items()))}")
    print(f"  Saved -> {SAMPLED_QUERIES_PATH}")
    return sampled


# ── CPB v4 inference, one ablation variant ────────────────────────────────────

def run_cpb_v4(queries: list[dict], ablation, variant_dir: Path) -> tuple[list[str], list[list[dict]]]:
    """Runs CPB v4 over `queries`, checkpointing one JSONL line per query to
    variant_dir/checkpoint.jsonl as it goes -- a 300-query run is dozens of
    LLM calls deep (risk scoring, PII analyzer, generation, SAD cascade,
    response guard) per query; without this, one bad query near the end of
    the loop would throw away every response computed before it, since
    responses.json/raw_chunks.json are otherwise only written once the
    whole variant finishes (see load_or_run_cpb)."""
    from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG
    from test_contre_mesure_ildpiltest._store import IldpilTestStore

    checkpoint_path = variant_dir / "checkpoint.jsonl"
    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done[row["global_id"]] = row
        print(f"  Resuming from checkpoint: {len(done)}/{len(queries)} queries already done")

    remaining = [q for q in queries if q["global_id"] not in done]

    if remaining:
        store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
        llm = LlamaLLM()
        naive_rag = NaiveRAG(store=store, llm=llm)
        cpb = CPBNaiveRAGV4(naive_rag=naive_rag, ablation=ablation)

        with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
            for i, q in enumerate(remaining):
                print(f"  CPB v4 [{ablation.name}] [{len(done) + i + 1}/{len(queries)}] {q['global_id']}...", end="\r")
                try:
                    result = cpb.run(get_query_text(q), top_k=TOP_K)
                    response = result["response"]
                    raw_chunks = result.get("raw_chunks", [])
                except Exception as exc:
                    response = f"ERROR: {exc}"
                    raw_chunks = []
                row = {"global_id": q["global_id"], "response": response, "raw_chunks": raw_chunks}
                done[q["global_id"]] = row
                ckpt_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                ckpt_f.flush()
        print()

    responses = [done[q["global_id"]]["response"] for q in queries]
    raw_chunks_per_query = [done[q["global_id"]]["raw_chunks"] for q in queries]
    checkpoint_path.unlink(missing_ok=True)
    return responses, raw_chunks_per_query


def load_or_run_cpb(
    queries: list[dict], ablation, variant_dir: Path, skip_generation: bool,
):
    """Returns (responses, raw_chunks_per_query, embedder)."""
    responses_path = variant_dir / "responses.json"
    chunks_path = variant_dir / "raw_chunks.json"

    if skip_generation and responses_path.exists() and chunks_path.exists():
        print(f"  Loading cached responses for {ablation.name}...")
        with open(responses_path, encoding="utf-8") as f:
            responses = json.load(f)
        with open(chunks_path, encoding="utf-8") as f:
            raw_chunks_per_query = json.load(f)
        print(f"  {len(responses)} responses loaded from cache.")
        from embeddings.embedder import Embedder
        return responses, raw_chunks_per_query, Embedder()

    print(f"  Running CPB v4 [{ablation.name}] (llama3.1:8b)...")
    variant_dir.mkdir(parents=True, exist_ok=True)
    responses, raw_chunks_per_query = run_cpb_v4(queries, ablation, variant_dir)
    from embeddings.embedder import Embedder
    embedder = Embedder()  # not reused from the store -- cheap to load, keeps cache/skip path uniform

    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(raw_chunks_per_query, f, ensure_ascii=False, indent=2)

    print(f"  Responses saved -> {responses_path}")
    return responses, raw_chunks_per_query, embedder


# ── Batched BERTScore — one call per variant instead of one per query ────────

def compute_bert_f1_batch(responses: list[str], raw_chunks_per_query: list[list[dict]]) -> list[float]:
    cand_idx = []
    cands, refs = [], []
    for i, (resp, chunks) in enumerate(zip(responses, raw_chunks_per_query)):
        if chunks and resp.strip():
            best_chunk = max(chunks, key=lambda c: c.get("similarity_score", 0))
            cands.append(resp)
            refs.append(best_chunk.get("text", ""))
            cand_idx.append(i)

    bert_f1 = [0.0] * len(responses)
    if not cands:
        return bert_f1

    from bert_score import score as bert_score_fn
    print(f"  BERTScore (batched, {len(cands)} pairs)...")
    _, _, F1 = bert_score_fn(cands, refs, lang="en", verbose=False)
    for idx, f1 in zip(cand_idx, F1.tolist()):
        bert_f1[idx] = float(f1)
    return bert_f1


# ── Phase 1: generation + metrics for one variant — NO mlflow here ───────────

def generate_and_score(ablation, queries: list[dict], skip_generation: bool) -> dict:
    variant_dir = OUT_ROOT / ablation.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    responses, raw_chunks_per_query, embedder = load_or_run_cpb(queries, ablation, variant_dir, skip_generation)

    print(f"  [{ablation.name}] [PII] ground-truth leakage...")
    from metrics.pii_leakage import compute_pii_leakage
    pii_path = variant_dir / "pii_results.json"
    if pii_path.exists():
        with open(pii_path, encoding="utf-8") as f:
            pii_dicts = json.load(f)
    else:
        pii_results = [
            compute_pii_leakage(response=resp, chunks=chunks, query=get_query_text(q))
            for resp, chunks, q in zip(responses, raw_chunks_per_query, queries)
        ]
        pii_dicts = [asdict(r) for r in pii_results]
        with open(pii_path, "w", encoding="utf-8") as f:
            json.dump(pii_dicts, f, ensure_ascii=False, indent=2)

    print(f"  [{ablation.name}] [Quality] exact match / ROUGE-L / BERTScore / answer relevancy...")
    from metrics.response_quality import compute_response_quality
    quality_path = variant_dir / "quality_results.json"
    if quality_path.exists():
        with open(quality_path, encoding="utf-8") as f:
            quality_dicts = json.load(f)
    else:
        bert_f1_batch = compute_bert_f1_batch(responses, raw_chunks_per_query)
        quality_results = [
            compute_response_quality(
                query=get_query_text(q),
                response=resp,
                chunks=chunks,
                target_entity=parse_target_entity(q.get("entity_hint", "")),
                embedder=embedder,
                precomputed_bert_f1=bf1,
            )
            for q, resp, chunks, bf1 in zip(queries, responses, raw_chunks_per_query, bert_f1_batch)
        ]
        quality_dicts = [asdict(r) for r in quality_results]
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(quality_dicts, f, ensure_ascii=False, indent=2)

    pii_leaked_total = sum(d["n_pii_leaked"] for d in pii_dicts)
    pii_total = sum(d["n_pii_total"] for d in pii_dicts)
    pii_rate = pii_leaked_total / pii_total if pii_total > 0 else 0.0
    n = len(quality_dicts)

    metrics = {
        "PII": pii_rate,
        "QS":  sum(d["quality_score"] for d in quality_dicts) / n,
        "AR":  sum(d["answer_relevancy"] for d in quality_dicts) / n,
        "RL":  sum(d["rouge_l"] for d in quality_dicts) / n,
        "BF1": sum(d["bert_score_f1"] for d in quality_dicts) / n,
        "EM":  sum(d["exact_match"] for d in quality_dicts) / n,
    }

    print(f"  [{ablation.name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # CSV export -- one row per query
    csv_path = variant_dir / "results_per_query.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "global_id", "query_id", "query_type", "query", "target_entity", "response",
            "pii_leaked", "pii_total", "pii_leakage_rate",
            "exact_match", "rouge_l", "bert_f1", "answer_relevancy", "quality_score",
        ])
        writer.writeheader()
        for q, resp, pii_d, q_d in zip(queries, responses, pii_dicts, quality_dicts):
            writer.writerow({
                "global_id":        q.get("global_id", ""),
                "query_id":         q.get("query_id", ""),
                "query_type":       q.get("query_type", ""),
                "query":            get_query_text(q)[:300],
                "target_entity":    parse_target_entity(q.get("entity_hint", "")) or "",
                "response":         resp,
                "pii_leaked":       pii_d["n_pii_leaked"],
                "pii_total":        pii_d["n_pii_total"],
                "pii_leakage_rate": round(pii_d["leakage_rate"], 4),
                "exact_match":      round(q_d["exact_match"], 4),
                "rouge_l":          round(q_d["rouge_l"], 4),
                "bert_f1":          round(q_d["bert_score_f1"], 4),
                "answer_relevancy": round(q_d["answer_relevancy"], 4),
                "quality_score":    round(q_d["quality_score"], 4),
            })

    variant_results = {
        "system":           "cpb_v4_ablation_cedh",
        "ablation_variant": ablation.name,
        "ablation_flags":   ablation.__dict__,
        "llm":              LLAMA_MODEL,
        "n_instances":      n,
        "metrics":          metrics,
        "metrics_detail": {
            "pii_leaked_total": pii_leaked_total,
            "pii_total":        pii_total,
        },
        "per_instance": {
            "PII":     pii_dicts,
            "QUALITY": quality_dicts,
        },
    }
    results_path = variant_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(variant_results, f, ensure_ascii=False, indent=2)

    return variant_results


# ── Phase 2: mlflow logging for every variant — called once, at the very end ──

def log_all_to_mlflow(queries: list[dict], variant_results: list[dict]) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for vr in variant_results:
        flags = vr["ablation_flags"]
        variant_dir = OUT_ROOT / vr["ablation_variant"]
        with mlflow.start_run(run_name=f"cpb_v4_ablation_cedh_{vr['ablation_variant']}"):
            mlflow.log_param("system", "cpb_v4")
            mlflow.log_param("ablation_variant", vr["ablation_variant"])
            for flag_name, flag_value in flags.items():
                if flag_name != "name":
                    mlflow.log_param(flag_name, flag_value)
            mlflow.log_param("llm_generation", LLAMA_MODEL)
            mlflow.log_param("dataset", "ildpil/text-anonymization-benchmark")
            mlflow.log_param("split", "test")
            mlflow.log_param("n_queries", len(queries))
            mlflow.log_param("n_responses", vr["n_instances"])
            mlflow.log_metric("pii_leaked_total", vr["metrics_detail"]["pii_leaked_total"])
            mlflow.log_metric("pii_total", vr["metrics_detail"]["pii_total"])
            for name, value in vr["metrics"].items():
                mlflow.log_metric(name, value)
            mlflow.log_artifact(str(variant_dir / "results.json"))
            mlflow.log_artifact(str(variant_dir / "results_per_query.csv"))


# ── Summary comparison across all 6 variants (full_pipeline = baseline) ──────

def print_and_save_summary(variant_results: list[dict]) -> None:
    baseline = next((vr for vr in variant_results if vr["ablation_variant"] == "full_pipeline"), None)
    baseline_metrics = baseline["metrics"] if baseline else None

    summary_path = OUT_ROOT / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["variant"] + METRIC_ORDER
        if baseline_metrics:
            header += [f"d_{m}" for m in METRIC_ORDER]
        writer.writerow(header)
        for vr in variant_results:
            m = vr["metrics"]
            row = [vr["ablation_variant"]] + [round(m[k], 4) for k in METRIC_ORDER]
            if baseline_metrics:
                row += [round(m[k] - baseline_metrics[k], 4) for k in METRIC_ORDER]
            writer.writerow(row)

    print("\n" + "=" * 100)
    print("  ABLATION SUMMARY -- CPB v4, CEDH/ECHR corpus (ildpil/text-anonymization-benchmark)")
    print("=" * 100)
    header = f"  {'Variant':<14}" + "".join(f"{m:>10}" for m in METRIC_ORDER)
    print(header)
    print("-" * 100)
    for vr in variant_results:
        m = vr["metrics"]
        print(f"  {vr['ablation_variant']:<14}" + "".join(f"{m[k]:>10.4f}" for k in METRIC_ORDER))
    print("=" * 100)
    for k in METRIC_ORDER:
        print(f"  {k:<4} = {METRIC_NAMES[k]}")
    print(f"  Saved -> {summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(skip_generation: bool = False, variant_names: list[str] | None = None,
          n_queries: int = 300, seed: int = 42):
    from countermeasure_v4.cpb_ablation import AblationConfig, VARIANTS  # spacy/torch load here, before chromadb/mlflow

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=== CPB v4 Ablation Study -- CEDH/ECHR corpus ===\n")

    all_variants = [AblationConfig(name="full_pipeline")] + VARIANTS
    variants = all_variants if not variant_names else [v for v in all_variants if v.name in variant_names]

    print("1. Loading query sample...")
    queries = load_sampled_queries(n_queries, seed)
    print(f"   {len(queries)} queries\n")

    # ── Phase 1: all ChromaDB/torch/BERTScore work, all variants. mlflow stays unimported. ──
    variant_results = []
    for i, ablation in enumerate(variants):
        print(f"2.{i + 1} Variant {ablation.name} ({i + 1}/{len(variants)})...")
        variant_results.append(generate_and_score(ablation, queries, skip_generation))
        print()

    # ── Phase 2: mlflow logging only, now that ChromaDB/torch are done for good. ──
    print("3. MLflow logging...")
    log_all_to_mlflow(queries, variant_results)

    print_and_save_summary(variant_results)
    print(f"\nDone. Per-variant results under {OUT_ROOT}\\<variant>\\results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPB v4 leave-one-out ablation study (CEDH/ECHR corpus)")
    parser.add_argument("--skip-generation", action="store_true",
                         help="Reuse cached responses per variant if present")
    parser.add_argument("--variants", nargs="+", default=None,
                         help="Run only these variants (default: all 6). "
                              "Choices: full_pipeline b0_off b1_b2_off b3_b4_off b6_off b7_off")
    parser.add_argument("--n-queries", type=int, default=300,
                         help="Size of the stratified query sample (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Sampling seed (default: 42)")
    args = parser.parse_args()
    main(skip_generation=args.skip_generation, variant_names=args.variants,
         n_queries=args.n_queries, seed=args.seed)
