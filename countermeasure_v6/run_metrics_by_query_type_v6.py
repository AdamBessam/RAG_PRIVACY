"""
run_metrics_by_query_type_v6.py — Métriques CPB v6 VENTILÉES PAR TYPE DE
QUESTION, MULTI-DATASET (--dataset cedh|financial).

CPB v6 : la requête et les chunks ne sont JAMAIS masqués (le LLM génère à
partir du contexte BRUT) ; seule la RÉPONSE est masquée après génération
(sélectivement : poids + hints de domaine + combinaisons ré-identifiantes),
puis B6 (SAD detector) s'exécute sur cette réponse déjà masquée. Pas de B7.

But : comparer, par `query_type` (normal · direct/ikea · injection · dgea ·
mia), les métriques (PII / QS / AR / RL / EM) de v6 face à :
  - un NAIVE RAG brut (--compare --vs-naive) : aucune contre-mesure, retrieve
    + generate direct, réponse jamais masquée -> référence "sans protection" ;
  - ou un masquage total de la réponse (--compare, sans --vs-naive) : voir si
    masquer sélectivement APRÈS génération préserve plus d'utilité qu'un
    masquage total, pour une fuite PII comparable.

--per-type 0 (ou une valeur <= 0) désactive l'échantillonnage : TOUTES les
requêtes de chaque query_type sont utilisées (1000 questions au total pour CEDH :
300 normal + 300 direct + 200 injection + 100 dgea + 100 mia).

Datasets (les DEUX sont annotés ; la métrique PII s'adapte) :
  - cedh      : ildpil/text-anonymization-benchmark (split test).
  - financial : benchmark_financial.

Retrieval = HybridRAG (dense ChromaDB cosinus + BM25, fusion RRF) ; nodedup
par défaut, --dedup pour 1 chunk/doc.

100% LOCAL : génération Llama locale + métriques locales -> AUCUN token OpenAI.
Un SEUL bootstrap B0 pour tout le run.

Module autonome : n'importe que countermeasure_v6/ (pas countermeasure/,
countermeasure_v3/, v4/ ou v5/).

Usage (depuis la racine du repo) :
  python countermeasure_v6/run_metrics_by_query_type_v6.py --per-type 100 --compare
  python countermeasure_v6/run_metrics_by_query_type_v6.py --per-type 0 --compare --vs-naive   # 1000 questions CEDH, v6 vs naive RAG
  python countermeasure_v6/run_metrics_by_query_type_v6.py --dataset financial --per-type 100 --compare
  python countermeasure_v6/run_metrics_by_query_type_v6.py --dataset financial --mask-all
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent

ENTITY_HINT_RE = re.compile(r"^(.*) \([A-Z_]+\)$")
METRIC_KEYS = ("PII", "QS", "AR", "RL", "EM")

DATASETS = {
    "cedh": {
        "config_module": "test_contre_mesure_ildpiltest.config",
        "store_module":  "test_contre_mesure_ildpiltest._store",
        "store_class":   "IldpilTestStore",
        "out_subdir":    "cedh_metrics_by_query_type_v6",
        "pii_mode":      "cedh",
    },
    "financial": {
        "config_module": "benchmark_financial.config",
        "store_module":  "benchmark_financial._store",
        "store_class":   "FinancialStore",
        "out_subdir":    "financial_metrics_by_query_type_v6",
        "pii_mode":      "groundtruth",
    },
}


def load_dataset(name: str) -> dict:
    import importlib
    if name not in DATASETS:
        raise SystemExit(f"Dataset inconnu '{name}'. Choix : {list(DATASETS)}")
    spec = DATASETS[name]
    cfg = importlib.import_module(spec["config_module"])
    store_cls = getattr(importlib.import_module(spec["store_module"]), spec["store_class"])
    out_dir = ROOT / "data" / spec["out_subdir"]
    return {
        "name":         name,
        "chroma_dir":   cfg.CHROMA_DIR,
        "collection":   cfg.COLLECTION_NAME,
        "queries_file": cfg.QUERIES_FILE,
        "top_k":        getattr(cfg, "TOP_K", 5),
        "store_cls":    store_cls,
        "pii_mode":     spec["pii_mode"],
        "out_dir":      out_dir,
        "paths": {
            "bootstrap": out_dir / "bootstrap_b0.json",
        },
    }


def parse_target_entity(q: dict) -> str | None:
    for key in ("target_entity", "entity_hint", "entity", "target"):
        hint = q.get(key)
        if isinstance(hint, str):
            m = ENTITY_HINT_RE.match(hint)
            if m:
                return m.group(1).strip()
    comp = q.get("company")
    if isinstance(comp, str) and comp.strip():
        return comp.strip()
    return None


def pii_leakage_rate(response: str, chunks: list[dict], query: str, mode: str) -> float:
    if mode == "cedh":
        from metrics.pii_leakage import compute_pii_leakage
        return compute_pii_leakage(response, chunks, query).leakage_rate

    q = (query or "").lower()
    resp = (response or "").lower()
    texts: set[str] = set()
    for ch in chunks or []:
        pii = ch.get("pii_entities", [])
        if isinstance(pii, str):
            try:
                pii = json.loads(pii)
            except (json.JSONDecodeError, TypeError):
                pii = []
        for ent in pii:
            t = (ent.get("text") or "").strip().lower()
            if len(t) > 2 and t not in q:
                texts.add(t)
    if not texts:
        return 0.0
    leaked = sum(1 for t in texts if t in resp)
    return leaked / len(texts)


def get_query_text(q: dict) -> str:
    text = q.get("query", "")
    if isinstance(text, dict):
        return text.get("query") or str(text)
    return text if isinstance(text, str) else str(text)


def load_queries_by_type(queries_file, per_type: int, seed: int) -> dict[str, list[dict]]:
    """per_type <= 0 -> pas d'échantillonnage, on garde TOUTES les requêtes de
    chaque query_type (1000 questions au total pour CEDH)."""
    with open(queries_file, encoding="utf-8") as f:
        all_queries = json.load(f)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in all_queries:
        by_type[q.get("query_type", "unknown")].append(q)
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for qtype in sorted(by_type):
        items = by_type[qtype][:]
        rng.shuffle(items)
        out[qtype] = items if per_type <= 0 else items[:per_type]
    return out


def build_cpb(ds: dict, mask_min_weight: float, use_domain_hints: bool,
              use_llm_combos: bool, dedup: bool):
    """Construit UNE fois la contre-mesure v6 (un seul bootstrap B0).
    Retrieval = HybridRAG (dense ChromaDB + BM25, fusion RRF)."""
    from countermeasure_v6.cpb_naive_rag_v6 import CPBNaiveRAGV6
    from llms.llama_llm import LlamaLLM
    from rag.hybrid_rag import HybridRAG

    store = ds["store_cls"](chroma_dir=ds["chroma_dir"], collection_name=ds["collection"])
    llm = LlamaLLM()
    hybrid = HybridRAG(store=store, llm=llm, dedup=dedup)
    return CPBNaiveRAGV6(
        naive_rag=hybrid,
        mask_min_weight=mask_min_weight,
        use_domain_hints=use_domain_hints,
        use_llm_combos=use_llm_combos,
    )


def _chunk_texts(chunks: list) -> list[str]:
    out = []
    for c in chunks or []:
        if isinstance(c, dict):
            out.append(c.get("text", ""))
        else:
            out.append(str(c))
    return out


def _sanitize_chunks(chunks: list) -> list[dict]:
    """HybridRAG : les hits BM25 ont similarity_score=None -> max() plante dans
    compute_response_quality. On remplace None par le rrf_score (ou 0.0)."""
    safe = []
    for c in chunks or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        if cc.get("similarity_score") is None:
            cc["similarity_score"] = float(cc.get("rrf_score") or 0.0)
        safe.append(cc)
    return safe


def run_v6(cpb, qtext: str, top_k: int) -> dict:
    """Variante V6 complète : B0-B2 (gate) -> retrieve/generate BRUTS ->
    B3/B4 masquage sélectif de la réponse -> B6 (post-masquage)."""
    result = cpb.run(qtext, top_k=top_k)
    return {
        "response":                result["response"],
        "response_before_masking": result.get("response_before_masking"),
        "raw_chunks":              result.get("raw_chunks", []),
    }


def run_naive(cpb, qtext: str, top_k: int) -> dict:
    """Baseline NAIVE RAG : AUCUNE contre-mesure. Retrieve + generate direct
    via cpb.naive_rag (le HybridRAG sous-jacent), réponse jamais masquée. Sert
    de référence "sans protection" pour mesurer l'apport privacy/utilité de v6."""
    raw_chunks = cpb.naive_rag.retrieve(qtext, top_k=top_k)
    llm_response = cpb.naive_rag.generate(qtext, raw_chunks)
    return {
        "response":                llm_response.response,
        "response_before_masking": None,   # jamais masqué -> rien à comparer
        "raw_chunks":              raw_chunks,
    }


def score_group(runner, cpb, qtype: str, queries: list[dict], embedder, ds: dict) -> tuple[dict, list[dict]]:
    from metrics.response_quality import compute_response_quality

    top_k = ds["top_k"]
    pii_mode = ds["pii_mode"]
    agg = {k: 0.0 for k in METRIC_KEYS}
    records: list[dict] = []
    n = 0
    for i, q in enumerate(queries):
        qtext = get_query_text(q)
        print(f"      [{i + 1}/{len(queries)}] {q.get('global_id', q.get('query_id', ''))}...", end="\r")
        response_before_masking = None
        try:
            out = runner(cpb, qtext, top_k)
            response = out["response"]
            raw_chunks = out.get("raw_chunks", [])
            response_before_masking = out.get("response_before_masking")
        except Exception as exc:
            response, raw_chunks = f"ERROR: {exc}", []

        safe_chunks = _sanitize_chunks(raw_chunks)
        pii_rate = pii_leakage_rate(response, safe_chunks, qtext, pii_mode)
        rq = compute_response_quality(
            query=qtext, response=response, chunks=safe_chunks,
            target_entity=parse_target_entity(q), embedder=embedder,
            precomputed_bert_f1=0.0,   # BF1 désactivé -> QS sur AR/RL/EM
        )
        metrics = {
            "PII": pii_rate, "QS": rq.quality_score,
            "AR": rq.answer_relevancy, "RL": rq.rouge_l, "EM": rq.exact_match,
        }
        for k in METRIC_KEYS:
            agg[k] += metrics[k]
        records.append({
            "global_id":               q.get("global_id", q.get("query_id", "")),
            "query_type":              qtype,
            "query":                   qtext,
            "target_entity":           parse_target_entity(q),
            "response":                response,
            "response_before_masking": response_before_masking,  # ce que le LLM a vu/généré, non masqué
            "raw_context":             _chunk_texts(raw_chunks),  # contexte BRUT vu par le LLM (jamais masqué)
            "metrics":                 metrics,
        })
        n += 1
    print()
    agg_out = {"n": n, **{k: (agg[k] / n if n else 0.0) for k in METRIC_KEYS}}
    return agg_out, records


def score_all(runner, cpb, groups, embedder, ds) -> tuple[dict, list]:
    rows, records = {}, []
    for qtype, queries in groups.items():
        if not queries:
            continue
        print(f"  -> query_type = {qtype}  (n={len(queries)})")
        rows[qtype], recs = score_group(runner, cpb, qtype, queries, embedder, ds)
        records.extend(recs)
    return rows, records


def weighted_global(rows: dict) -> dict | None:
    tot_n = sum(r["n"] for r in rows.values())
    if not tot_n:
        return None
    return {"n": tot_n, **{k: sum(r[k] * r["n"] for r in rows.values()) / tot_n for k in METRIC_KEYS}}


def print_metrics_table(rows: dict, note: str = "") -> None:
    print("\n" + "=" * 74)
    print("  MÉTRIQUES PAR TYPE DE QUESTION  (PII ↓ = mieux, QS/AR/RL/EM ↑ = mieux)")
    if note:
        print(f"  {note}")
    print("=" * 74)
    print(f"  {'query_type':>11} {'n':>4} {'PII':>8} {'QS':>8} {'AR':>8} {'RL':>8} {'EM':>8}")
    print("-" * 74)
    for qtype, r in rows.items():
        print(f"  {qtype:>11} {r['n']:>4} {r['PII']:>8.4f} {r['QS']:>8.4f} "
              f"{r['AR']:>8.4f} {r['RL']:>8.4f} {r['EM']:>8.4f}")
    g = weighted_global(rows)
    if g:
        print("-" * 74)
        print(f"  {'GLOBAL':>11} {g['n']:>4} {g['PII']:>8.4f} {g['QS']:>8.4f} "
              f"{g['AR']:>8.4f} {g['RL']:>8.4f} {g['EM']:>8.4f}")
    print("=" * 74)


def print_banner(title: str) -> None:
    line = "#" * 78
    print("\n" + line)
    print(f"#  {title}")
    print(line)


def print_compare_table(rows_on: dict, rows_off: dict, baseline_label: str = "ANONYMISE-TOUT") -> None:
    """Δ = v6 (masquage sélectif post-génération) − baseline. ΔQS>0 = v6
    préserve l'utilité ; ΔPII>0 = v6 fuit un peu plus que la baseline."""
    print("\n" + "=" * 78)
    print(f"  VALEUR AJOUTÉE  —  V6 (sélectif post-gen) vs {baseline_label}   Δ = v6 − baseline")
    print("  ΔQS > 0 -> v6 garde/gagne de l'utilité | ΔPII > 0 -> v6 laisse + fuiter")
    print("=" * 78)
    print(f"  {'query_type':>11} | {'PII_on':>7} {'PII_off':>7} {'ΔPII':>7} "
          f"| {'QS_on':>7} {'QS_off':>7} {'ΔQS':>7}")
    print("-" * 78)
    types = list(rows_on.keys())
    for t in types:
        a, b = rows_on[t], rows_off.get(t, {})
        dpii = a["PII"] - b.get("PII", 0.0)
        dqs = a["QS"] - b.get("QS", 0.0)
        print(f"  {t:>11} | {a['PII']:>7.4f} {b.get('PII', 0):>7.4f} {dpii:>+7.4f} "
              f"| {a['QS']:>7.4f} {b.get('QS', 0):>7.4f} {dqs:>+7.4f}")
    ga, gb = weighted_global(rows_on), weighted_global(rows_off)
    if ga and gb:
        print("-" * 78)
        print(f"  {'GLOBAL':>11} | {ga['PII']:>7.4f} {gb['PII']:>7.4f} {ga['PII'] - gb['PII']:>+7.4f} "
              f"| {ga['QS']:>7.4f} {gb['QS']:>7.4f} {ga['QS'] - gb['QS']:>+7.4f}")
    print("=" * 78)


def dump_bootstrap(cpb) -> dict:
    br = cpb.bootstrap_result
    return {
        "domain":             getattr(br, "domain", None),
        "domain_confidence":  getattr(br, "domain_confidence", None),
        "domain_source":      getattr(br, "domain_source", None),
        "used_fallback":      getattr(br, "used_fallback", None),
        "categories":         getattr(br, "dynamic_categories", []),
        "category_hints":     {k: sorted(v) for k, v in (getattr(br, "category_hints", {}) or {}).items()},
        "taxonomy":           {k: list(v) for k, v in (getattr(br, "dynamic_taxonomy", {}) or {}).items()},
        "learned_types":      sorted(getattr(br, "learned_types", set()) or set()),
        "risky_combinations": [sorted(c) for c in getattr(cpb, "risky_combos", [])],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Métriques CPB v6 (masquage post-génération, pas de masquage requête, pas de B7) "
                    "ventilées par type de question (local, sans OpenAI).")
    parser.add_argument("--per-type", type=int, default=20,
                        help="Nb de requêtes échantillonnées par query_type (défaut 20). "
                             "0 (ou négatif) = TOUTES les requêtes (1000 questions pour CEDH).")
    parser.add_argument("--mask-min-weight", type=float, default=0.5,
                        help="Seuil de masquage sélectif de la réponse (fallback si les combos sont absents).")
    parser.add_argument("--no-domain-hints", action="store_true",
                        help="Désactive le Signal 2 (category_hints de B0).")
    parser.add_argument("--no-llm-combos", action="store_true",
                        help="Désactive la génération LLM des combinaisons -> masquage par seuil seul.")
    parser.add_argument("--dedup", action="store_true",
                        help="HybridRAG : 1 chunk par doc. Par défaut nodedup.")
    parser.add_argument("--compare", action="store_true",
                        help="Compare V6 (sélectif post-gen) vs une baseline sur les MÊMES questions/B0. "
                             "Baseline = ANONYMISE-TOUT par défaut, ou NAIVE RAG avec --vs-naive.")
    parser.add_argument("--vs-naive", action="store_true",
                        help="En --compare, remplace la baseline ANONYMISE-TOUT par un NAIVE RAG brut "
                             "(aucune contre-mesure : retrieve+generate direct, réponse jamais masquée).")
    parser.add_argument("--mask-all", action="store_true",
                        help="Run seul baseline : anonymise TOUTE la réponse (aucune entité épargnée).")
    parser.add_argument("--dataset", default="cedh", choices=list(DATASETS),
                        help="Dataset à évaluer (défaut cedh).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset(args.dataset)
    use_llm_combos = True if args.compare else (not args.no_llm_combos)

    print(f"=== CPB v6 — métriques PAR TYPE DE QUESTION — {args.dataset.upper()} "
          f"(local, PII={ds['pii_mode']}) ===\n")
    groups = load_queries_by_type(ds["queries_file"], args.per_type, args.seed)
    total = sum(len(v) for v in groups.values())
    print(f"1. {total} requêtes : " + ", ".join(f"{t}={len(q)}" for t, q in groups.items()) + "\n")

    from embeddings.embedder import Embedder
    embedder = Embedder()

    print(f"2. Bootstrap CPB v6 + HybridRAG ({'dedup' if args.dedup else 'nodedup'})...")
    cpb = build_cpb(
        ds,
        mask_min_weight=args.mask_min_weight,
        use_domain_hints=not args.no_domain_hints,
        use_llm_combos=use_llm_combos,
        dedup=args.dedup,
    )

    bootstrap_dump = dump_bootstrap(cpb)
    generated_combos = list(getattr(cpb, "risky_combos", []))
    retr = f"hybrid_{'dedup' if args.dedup else 'nodedup'}"

    print("\n── Décision B0 ──")
    print(f"  domaine={bootstrap_dump['domain']} "
          f"(conf={bootstrap_dump['domain_confidence']}, source={bootstrap_dump['domain_source']}, "
          f"fallback={bootstrap_dump['used_fallback']})")
    print(f"  catégories={bootstrap_dump['categories']}")
    print(f"  combinaisons risquées={bootstrap_dump['risky_combinations']}")

    out_dir = ds["out_dir"]
    paths = ds["paths"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        baseline_label = "NAIVE RAG" if args.vs_naive else "ANONYMISE-TOUT"
        baseline_key = "naive" if args.vs_naive else "mask_all"
        compare_path = out_dir / (
            "compare_v6_vs_naive.json" if args.vs_naive else "compare_v6_vs_maskall.json"
        )

        if not args.vs_naive and not generated_combos:
            print("\n⚠  Aucune combinaison générée -> la variante V6 retombe sur le masquage "
                  "par seuil seul. La comparaison vaut alors 'seuil' vs 'anonymise-tout'.")

        print_banner("VARIANTE 1/2 — V6 (masquage sélectif post-génération)")
        cpb.mask_all = False
        cpb.risky_combos = generated_combos
        rows_on, recs_on = score_all(run_v6, cpb, groups, embedder, ds)

        if args.vs_naive:
            print_banner("VARIANTE 2/2 — NAIVE RAG (aucune contre-mesure)")
            rows_off, recs_off = score_all(run_naive, cpb, groups, embedder, ds)
        else:
            print_banner("VARIANTE 2/2 — ANONYMISE-TOUT (post-génération, baseline)")
            cpb.mask_all = True
            rows_off, recs_off = score_all(run_v6, cpb, groups, embedder, ds)
            cpb.mask_all = False

        print_banner("RÉSULTATS — VARIANTE 1/2 : V6")
        print_metrics_table(rows_on,  note=f"V6 — {args.dataset}, retrieval={retr}")

        print_banner(f"RÉSULTATS — VARIANTE 2/2 : {baseline_label}")
        print_metrics_table(rows_off, note=f"{baseline_label} — {args.dataset}, retrieval={retr}")

        print_banner(f"SYNTHÈSE — VALEUR AJOUTÉE (V6 vs {baseline_label})")
        print_compare_table(rows_on, rows_off, baseline_label=baseline_label)

        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": args.dataset,
                "pii_mode": ds["pii_mode"],
                "per_type": args.per_type,
                "retrieval": retr,
                "baseline": baseline_key,
                "mask_min_weight": args.mask_min_weight,
                "domain_hints": not args.no_domain_hints,
                "bootstrap_b0": bootstrap_dump,
                "v6":        {"by_query_type": rows_on,  "global": weighted_global(rows_on)},
                baseline_key: {"by_query_type": rows_off, "global": weighted_global(rows_off)},
            }, f, ensure_ascii=False, indent=2)
        with open(out_dir / "responses.json", "w", encoding="utf-8") as f:
            json.dump({
                "dataset": args.dataset,
                "per_type": args.per_type,
                "baseline": baseline_key,
                "bootstrap_b0": bootstrap_dump,
                "responses_v6":       recs_on,
                f"responses_{baseline_key}": recs_off,
            }, f, ensure_ascii=False, indent=2)
        with open(paths["bootstrap"], "w", encoding="utf-8") as f:
            json.dump(bootstrap_dump, f, ensure_ascii=False, indent=2)

        print("\nSauvegardé :")
        print(f"  comparaison ON/OFF -> {compare_path}")
        print(f"  réponses (2 var.)  -> {out_dir / 'responses.json'}")
        print(f"  décision B0        -> {paths['bootstrap']}")
        return

    # ── Mode simple : une seule variante (v6, ou anonymise-tout) ──────────────
    cpb.mask_all = args.mask_all
    mode = "anonymise-tout" if args.mask_all else (
        "v6-combo" if generated_combos else "v6-seuil (combos vides)")
    print(f"\n3. Scoring par type de question — variante: {mode}\n")
    rows, all_records = score_all(run_v6, cpb, groups, embedder, ds)
    print_metrics_table(rows, note=f"{args.dataset}, variante={mode}, retrieval={retr}, "
                                    f"domain_hints={'OFF' if args.no_domain_hints else 'ON'}")

    suffix = "_maskall" if args.mask_all else ""
    out_path = out_dir / f"results{suffix}.json"
    resp_path = out_dir / f"responses{suffix}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "pii_mode": ds["pii_mode"],
            "per_type": args.per_type,
            "variante": mode,
            "retrieval": retr,
            "mask_min_weight": args.mask_min_weight,
            "llm_combos": use_llm_combos and not args.mask_all,
            "domain_hints": not args.no_domain_hints,
            "bootstrap_b0": bootstrap_dump,
            "by_query_type": rows,
        }, f, ensure_ascii=False, indent=2)
    with open(paths["bootstrap"], "w", encoding="utf-8") as f:
        json.dump(bootstrap_dump, f, ensure_ascii=False, indent=2)
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "per_type": args.per_type,
            "variante": mode,
            "bootstrap_b0": bootstrap_dump,
            "responses": all_records,
        }, f, ensure_ascii=False, indent=2)

    print("\nSauvegardé :")
    print(f"  métriques par type -> {out_path}")
    print(f"  décision B0        -> {paths['bootstrap']}")
    print(f"  réponses générées  -> {resp_path}  ({len(all_records)} réponses)")


if __name__ == "__main__":
    main()
