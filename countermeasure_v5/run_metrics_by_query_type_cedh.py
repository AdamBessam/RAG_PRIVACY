"""
run_metrics_by_query_type_cedh.py — Métriques CPB v5 (combo) VENTILÉES PAR TYPE DE
QUESTION, MULTI-DATASET (--dataset cedh|financial).

But : ne PAS agréger toutes les requêtes ensemble, mais comparer les métriques
(PII / QS / AR / RL / EM) SÉPARÉMENT pour chaque `query_type` du corpus
(normal · direct/ikea · injection · dgea · mia) → on voit quel type de question
réagit comment à la contre-mesure.

Datasets (les DEUX sont annotés ; la métrique PII s'adapte) :
  - cedh      : ildpil/text-anonymization-benchmark (split test). PII = entités
                SENSIBLES (sensitivity ildpil) via compute_pii_leakage.
  - financial : benchmark_financial. PII = TOUTES les PII annotées des chunks
                (ground-truth), exclusion de celles déjà dans la question.

Retrieval = HybridRAG (dense ChromaDB cosinus + BM25, fusion RRF) ; par défaut
nodedup, --dedup pour 1 chunk/doc.
Masquage = CPB v5 combo (COMBINAISONS ré-identifiantes générées par B0/Llama pour
le domaine détecté ; on masque pour casser toute combinaison présente ;
identifiants forts toujours masqués). --mask-all = baseline anonymise-tout,
--compare = combo vs anonymise-tout sur les mêmes questions.

100 % LOCAL : génération Llama locale + métriques locales → AUCUN token OpenAI.
Un SEUL bootstrap B0 pour tout le run.

Usage (depuis la racine du repo) :
  python countermeasure_v5/run_metrics_by_query_type_cedh.py --per-type 100 --compare
  python countermeasure_v5/run_metrics_by_query_type_cedh.py --dataset financial --per-type 100 --compare
  python countermeasure_v5/run_metrics_by_query_type_cedh.py --dataset financial --mask-all
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

# Registre des datasets : config + store + métrique PII adaptée + dossier de sortie.
DATASETS = {
    "cedh": {
        "config_module": "test_contre_mesure_ildpiltest.config",
        "store_module":  "test_contre_mesure_ildpiltest._store",
        "store_class":   "IldpilTestStore",
        "out_subdir":    "cedh_metrics_by_query_type",
        "pii_mode":      "cedh",        # PII sensibles (sensitivity ildpil)
    },
    "financial": {
        "config_module": "benchmark_financial.config",
        "store_module":  "benchmark_financial._store",
        "store_class":   "FinancialStore",
        "out_subdir":    "financial_metrics_by_query_type",
        "pii_mode":      "groundtruth", # toutes les PII annotées (pas de sensitivity)
    },
}


def load_dataset(name: str) -> dict:
    """Charge dynamiquement config + store + paramètres du dataset choisi."""
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
            "results":   out_dir / "results.json",
            "responses": out_dir / "responses.json",
            "bootstrap": out_dir / "bootstrap_b0.json",
            "compare":   out_dir / "compare_combo_vs_maskall.json",
        },
    }


def parse_target_entity(q: dict) -> str | None:
    # 1) hint structuré "texte (TYPE)" (CEDH)
    for key in ("target_entity", "entity_hint", "entity", "target"):
        hint = q.get(key)
        if isinstance(hint, str):
            m = ENTITY_HINT_RE.match(hint)
            if m:
                return m.group(1).strip()
    # 2) champ 'company' en clair (financier) → cible de l'EM
    comp = q.get("company")
    if isinstance(comp, str) and comp.strip():
        return comp.strip()
    return None


def pii_leakage_rate(response: str, chunks: list[dict], query: str, mode: str) -> float:
    """Taux de fuite PII, adapté au dataset (les DEUX sont annotés) :
      - mode 'cedh'        : entités SENSIBLES (sensitivity ildpil), via compute_pii_leakage.
      - mode 'groundtruth' : TOUTES les PII annotées des chunks (financier), matching exact
                             sur le texte, en excluant les PII déjà présentes dans la question."""
    if mode == "cedh":
        from metrics.pii_leakage import compute_pii_leakage
        return compute_pii_leakage(response, chunks, query).leakage_rate

    # Ground-truth : toutes les PII annotées, exclusion de celles déjà dans la question.
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
            if len(t) > 2 and t not in q:      # exclut les PII déjà connues de l'utilisateur
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
    """Échantillonne `per_type` requêtes PAR query_type depuis le corpus 1000."""
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
        out[qtype] = items[:per_type]
    return out


def build_cpb(ds: dict, mask_min_weight: float, use_domain_hints: bool,
              use_llm_combos: bool, dedup: bool):
    """Construit UNE fois la contre-mesure (un seul bootstrap B0).
    Retrieval = HybridRAG (dense ChromaDB + BM25, fusion RRF). Store selon le dataset."""
    from countermeasure_v5.cpb_naive_rag_v5_combo import CPBNaiveRAGV5Combo
    from llms.llama_llm import LlamaLLM
    from rag.hybrid_rag import HybridRAG

    store = ds["store_cls"](chroma_dir=ds["chroma_dir"], collection_name=ds["collection"])
    llm = LlamaLLM()
    # HybridRAG : dense + BM25 + RRF. dedup=False → plusieurs chunks/doc (config hybrid_nodedup).
    hybrid = HybridRAG(store=store, llm=llm, dedup=dedup)
    return CPBNaiveRAGV5Combo(
        naive_rag=hybrid,
        mask_min_weight=mask_min_weight,
        use_domain_hints=use_domain_hints,
        use_llm_combos=use_llm_combos,
    )


def _chunk_texts(chunks: list) -> list[str]:
    """Texte de chaque chunk (masqué si dict CPB, sinon str brut)."""
    out = []
    for c in chunks or []:
        if isinstance(c, dict):
            out.append(c.get("text", ""))
        else:
            out.append(str(c))
    return out


def _sanitize_chunks(chunks: list) -> list[dict]:
    """HybridRAG : les hits BM25 ont similarity_score=None → max() plante dans
    compute_response_quality. On remplace None par le rrf_score (ou 0.0), sur une
    COPIE, sans muter les chunks d'origine ni toucher au code métrique partagé."""
    safe = []
    for c in chunks or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        if cc.get("similarity_score") is None:
            cc["similarity_score"] = float(cc.get("rrf_score") or 0.0)
        safe.append(cc)
    return safe


def score_group(cpb, qtype: str, queries: list[dict], embedder, ds: dict) -> tuple[dict, list[dict]]:
    """Génère + score un groupe de requêtes (un query_type).
    Renvoie (métriques agrégées, liste des enregistrements par requête)."""
    from metrics.response_quality import compute_response_quality

    top_k = ds["top_k"]
    pii_mode = ds["pii_mode"]
    agg = {k: 0.0 for k in METRIC_KEYS}
    records: list[dict] = []
    n = 0
    for i, q in enumerate(queries):
        qtext = get_query_text(q)
        print(f"      [{i + 1}/{len(queries)}] {q.get('global_id', q.get('query_id', ''))}...", end="\r")
        masked_query, masked_context = qtext, []
        try:
            result = cpb.run(qtext, top_k=top_k)
            response = result["response"]
            raw_chunks = result.get("raw_chunks", [])
            masked_query = result.get("cpb_masked_query", qtext)
            masked_context = _chunk_texts(result.get("chunks", []))
        except Exception as exc:
            response, raw_chunks = f"ERROR: {exc}", []

        safe_chunks = _sanitize_chunks(raw_chunks)
        pii_rate = pii_leakage_rate(response, safe_chunks, qtext, pii_mode)
        rq = compute_response_quality(
            query=qtext, response=response, chunks=safe_chunks,
            target_entity=parse_target_entity(q), embedder=embedder,
            precomputed_bert_f1=0.0,   # BF1 désactivé → QS sur AR/RL/EM
        )
        metrics = {
            "PII": pii_rate, "QS": rq.quality_score,
            "AR": rq.answer_relevancy, "RL": rq.rouge_l, "EM": rq.exact_match,
        }
        for k in METRIC_KEYS:
            agg[k] += metrics[k]
        records.append({
            "global_id":      q.get("global_id", q.get("query_id", "")),
            "query_type":     qtype,
            "query":          qtext,
            "target_entity":  parse_target_entity(q),
            "masked_query":   masked_query,
            "response":       response,
            "masked_context": masked_context,   # ce que le LLM a réellement vu (après masquage)
            "metrics":        metrics,
        })
        n += 1
    print()
    agg_out = {"n": n, **{k: (agg[k] / n if n else 0.0) for k in METRIC_KEYS}}
    return agg_out, records


def score_all(cpb, groups, embedder, ds) -> tuple[dict, list]:
    """Score tous les query_type ; renvoie (rows par type, tous les records)."""
    rows, records = {}, []
    for qtype, queries in groups.items():
        if not queries:
            continue
        print(f"  -> query_type = {qtype}  (n={len(queries)})")
        rows[qtype], recs = score_group(cpb, qtype, queries, embedder, ds)
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
    """Grande bannière pour séparer nettement chaque variante à l'écran."""
    line = "#" * 78
    print("\n" + line)
    print(f"#  {title}")
    print(line)


def print_compare_table(rows_on: dict, rows_off: dict) -> None:
    """Comparaison COMBO vs ANONYMISE-TOUT, par type. Δ = combo − mask_all.
    ΔQS>0 = les combos préservent l'utilité vs tout masquer ;
    ΔPII>0 = les combos laissent (un peu) plus fuiter que tout masquer."""
    print("\n" + "=" * 78)
    print("  VALEUR AJOUTÉE  —  COMBO vs ANONYMISE-TOUT   Δ = combo − mask_all")
    print("  ΔQS > 0 → combos gardent l'utilité | ΔPII > 0 → combos laissent + fuiter")
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
        description="Métriques CPB v5 contextuel ventilées par type de question (CEDH, local, sans OpenAI).")
    parser.add_argument("--per-type", type=int, default=20,
                        help="Nb de requêtes échantillonnées par query_type (défaut 20).")
    parser.add_argument("--mask-min-weight", type=float, default=0.5,
                        help="Seuil v5 (utilisé par le fallback si les combos sont absents).")
    parser.add_argument("--no-domain-hints", action="store_true",
                        help="Désactive le Signal 2 (category_hints de B0).")
    parser.add_argument("--no-llm-combos", action="store_true",
                        help="Désactive la génération LLM des combinaisons → fallback masquage v5.")
    parser.add_argument("--dedup", action="store_true",
                        help="HybridRAG : 1 chunk par doc. Par défaut nodedup (plusieurs chunks/doc).")
    parser.add_argument("--compare", action="store_true",
                        help="Compare COMBO vs ANONYMISE-TOUT sur les MÊMES questions/B0 → valeur ajoutée.")
    parser.add_argument("--mask-all", action="store_true",
                        help="Run seul baseline : anonymise TOUT (aucune entité épargnée), sorties *_maskall.json.")
    parser.add_argument("--dataset", default="cedh", choices=list(DATASETS),
                        help="Dataset à évaluer (défaut cedh). Détermine config/store/métrique PII.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset(args.dataset)

    # En mode compare on a besoin des combos générés (la variante ON).
    use_llm_combos = True if args.compare else (not args.no_llm_combos)

    print(f"=== CPB v5 combo — métriques PAR TYPE DE QUESTION — {args.dataset.upper()} "
          f"(local, PII={ds['pii_mode']}) ===\n")
    groups = load_queries_by_type(ds["queries_file"], args.per_type, args.seed)
    total = sum(len(v) for v in groups.values())
    print(f"1. {total} requêtes : " + ", ".join(f"{t}={len(q)}" for t, q in groups.items()) + "\n")

    from embeddings.embedder import Embedder
    embedder = Embedder()

    print(f"2. Bootstrap CPB v5 combo + HybridRAG "
          f"({'dedup' if args.dedup else 'nodedup'})...")
    cpb = build_cpb(
        ds,
        mask_min_weight=args.mask_min_weight,
        use_domain_hints=not args.no_domain_hints,
        use_llm_combos=use_llm_combos,
        dedup=args.dedup,
    )

    # ── Décision B0 (commune aux deux variantes) ─────────────────────────────
    bootstrap_dump = dump_bootstrap(cpb)
    generated_combos = list(getattr(cpb, "risky_combos", []))
    retr = f"hybrid_{'dedup' if args.dedup else 'nodedup'}"

    print("\n── Décision B0 ──")
    print(f"  domaine={bootstrap_dump['domain']} "
          f"(conf={bootstrap_dump['domain_confidence']}, source={bootstrap_dump['domain_source']}, "
          f"fallback={bootstrap_dump['used_fallback']})")
    print(f"  catégories={bootstrap_dump['categories']}")
    print(f"  category_hints={bootstrap_dump['category_hints']}")
    print(f"  combinaisons risquées={bootstrap_dump['risky_combinations']}")

    out_dir = ds["out_dir"]
    paths = ds["paths"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # ── Comparaison : MÊMES questions/B0/retrieval, combos ON puis OFF ─────
        if not generated_combos:
            print("\n⚠  Aucune combinaison générée → la variante 'COMBO' retombe sur le "
                  "masquage sélectif v5 (pas de vraie combinaison). La comparaison vaut "
                  "alors 'v5 sélectif' vs 'anonymise-tout'.")

        print_banner("VARIANTE 1/2 — COMBO (combinaisons LLM actives)")
        cpb.mask_all = False
        cpb.risky_combos = generated_combos
        rows_on, recs_on = score_all(cpb, groups, embedder, ds)

        print_banner("VARIANTE 2/2 — ANONYMISE-TOUT (baseline, on masque tout)")
        cpb.mask_all = True            # baseline : aucune entité épargnée
        rows_off, recs_off = score_all(cpb, groups, embedder, ds)
        cpb.mask_all = False

        # ── Métriques de chaque variante, séparées par une bannière ───────────
        print_banner("RÉSULTATS — VARIANTE 1/2 : COMBO")
        print_metrics_table(rows_on,  note=f"COMBO — {args.dataset}, retrieval={retr}")

        print_banner("RÉSULTATS — VARIANTE 2/2 : ANONYMISE-TOUT")
        print_metrics_table(rows_off, note=f"ANONYMISE-TOUT — {args.dataset}, retrieval={retr}")

        print_banner("SYNTHÈSE — VALEUR AJOUTÉE (COMBO vs ANONYMISE-TOUT)")
        print_compare_table(rows_on, rows_off)

        with open(paths["compare"], "w", encoding="utf-8") as f:
            json.dump({
                "dataset": args.dataset,
                "pii_mode": ds["pii_mode"],
                "per_type": args.per_type,
                "retrieval": retr,
                "mask_min_weight": args.mask_min_weight,
                "domain_hints": not args.no_domain_hints,
                "bootstrap_b0": bootstrap_dump,
                "combo":     {"by_query_type": rows_on,  "global": weighted_global(rows_on)},
                "mask_all":  {"by_query_type": rows_off, "global": weighted_global(rows_off)},
            }, f, ensure_ascii=False, indent=2)
        with open(paths["responses"], "w", encoding="utf-8") as f:
            json.dump({
                "dataset": args.dataset,
                "per_type": args.per_type,
                "bootstrap_b0": bootstrap_dump,
                "responses_combo":    recs_on,
                "responses_mask_all": recs_off,
            }, f, ensure_ascii=False, indent=2)
        with open(paths["bootstrap"], "w", encoding="utf-8") as f:
            json.dump(bootstrap_dump, f, ensure_ascii=False, indent=2)

        print("\nSauvegardé :")
        print(f"  comparaison ON/OFF → {paths['compare']}")
        print(f"  réponses (2 var.)  → {paths['responses']}")
        print(f"  décision B0        → {paths['bootstrap']}")
        return

    # ── Mode simple : une seule variante (combo, ou anonymise-tout) ───────────
    cpb.mask_all = args.mask_all
    mode = "anonymise-tout" if args.mask_all else (
        "combo" if generated_combos else "v5-sélectif (combos vides)")
    print(f"\n3. Scoring par type de question — variante: {mode}\n")
    rows, all_records = score_all(cpb, groups, embedder, ds)
    print_metrics_table(rows, note=f"{args.dataset}, variante={mode}, retrieval={retr}, "
                                    f"domain_hints={'OFF' if args.no_domain_hints else 'ON'}")

    # Sorties dédiées pour le baseline afin de ne pas écraser le run combo.
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
    print(f"  métriques par type → {out_path}")
    print(f"  décision B0        → {paths['bootstrap']}")
    print(f"  réponses générées  → {resp_path}  ({len(all_records)} réponses)")


if __name__ == "__main__":
    main()
