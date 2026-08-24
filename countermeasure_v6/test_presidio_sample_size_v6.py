"""
Test manuel — impact de la taille d'échantillon (n) sur la découverte PII via
Presidio (étape 0a de CPBBootstrapV6). Compare n=30/40/50/60 : types PII
détectés, occurrences, couverture documentaire et temps d'exécution, puis
conclut si n=50 est un choix raisonnable pour ce corpus.

Affiche tout au terminal ET enregistre les résultats détaillés dans un JSON.

Usage:
    python countermeasure_v6/test_presidio_sample_size_v6.py [chroma_dir] [collection_name] [output_json]

Defaults:
    chroma_dir      = data/chroma_chatdoctor
    collection_name = chatdoctor_eval_corpus
    output_json     = countermeasure_v6/test_presidio_sample_size_results.json
"""
import io
import json
import logging
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows to avoid emoji/accent encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)  # silence les logs tiers (presidio, spacy...)

import chromadb
from chromadb.config import Settings

# Import direct du module (pas du package countermeasure_v6) pour éviter de
# charger __init__.py, qui importe cpb_naive_rag_v6 -> cpb_pii_v6/cpb_sad_detector_v6
# (Presidio/GLiNER/torch) inutilement ici. Même technique que
# countermeasure_v4/test_bootstrap_v4.py.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "cpb_bootstrap_v6",
    str(Path(__file__).parent / "cpb_bootstrap_v6.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CPBBootstrapV6 = _mod.CPBBootstrapV6

SAMPLE_SIZES = [30, 40, 50, 60, 70, 80, 90]


def sep(title: str, char: str = "="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(char * width)


class _RawCollectionStore:
    """Wrapper minimal exposant .collection, comme attendu par CPBBootstrapV6."""

    def __init__(self, chroma_dir: str, collection_name: str):
        client = chromadb.PersistentClient(
            path=chroma_dir, settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(collection_name)


def run_presidio_pass(bootstrap: "CPBBootstrapV6", n: int) -> dict:
    """Reproduit exactement _step_0a (sample aléatoire + Presidio, doc[:2000]),
    mais avec un n paramétrable et des compteurs détaillés par type."""
    from presidio_analyzer import AnalyzerEngine

    t0 = time.perf_counter()
    chunks = bootstrap._sample_chunks(n)

    analyzer = AnalyzerEngine()
    occurrences: dict[str, int] = {}
    docs_with_type: dict[str, int] = {}

    for doc in chunks:
        if not doc:
            continue
        seen_in_doc = set()
        for finding in analyzer.analyze(text=doc[:2000], language="en"):
            t = finding.entity_type.upper()
            occurrences[t] = occurrences.get(t, 0) + 1
            seen_in_doc.add(t)
        for t in seen_in_doc:
            docs_with_type[t] = docs_with_type.get(t, 0) + 1

    elapsed = time.perf_counter() - t0

    return {
        "n_requested": n,
        "n_chunks_sampled": len(chunks),
        "elapsed_seconds": round(elapsed, 2),
        "types": {
            t: {"occurrences": occurrences[t], "docs_with_type": docs_with_type.get(t, 0)}
            for t in sorted(occurrences)
        },
    }


def print_pass_result(n: int, result: dict):
    sep(f"n = {n} — Presidio sur {result['n_chunks_sampled']} chunk(s) échantillonné(s)")
    print(f"  Temps d'exécution : {result['elapsed_seconds']}s")
    print(f"  Types PII trouvés : {len(result['types'])}\n")
    for t, stats in sorted(result["types"].items(), key=lambda kv: -kv[1]["occurrences"]):
        pct_docs = 100 * stats["docs_with_type"] / result["n_chunks_sampled"] if result["n_chunks_sampled"] else 0
        print(
            f"    • {t:<20} {stats['occurrences']:>4} occurrence(s)  "
            f"— présent dans {stats['docs_with_type']}/{result['n_chunks_sampled']} chunks ({pct_docs:.0f}%)"
        )


def main():
    chroma_dir = sys.argv[1] if len(sys.argv) > 1 else "data/chroma_chatdoctor"
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "chatdoctor_eval_corpus"
    output_json = sys.argv[3] if len(sys.argv) > 3 else str(
        Path(__file__).parent / "test_presidio_sample_size_results.json"
    )

    sep("CONNEXION CHROMADB")
    store = _RawCollectionStore(chroma_dir, collection_name)
    total_chunks = store.collection.count()
    print(f"  Collection          : {collection_name}")
    print(f"  Chunks disponibles  : {total_chunks}")

    bootstrap = CPBBootstrapV6(store=store)

    results: dict[int, dict] = {}
    for n in SAMPLE_SIZES:
        result = run_presidio_pass(bootstrap, n)
        print_pass_result(n, result)
        results[n] = result

    # ── Comparaison entre tailles ────────────────────────────────────────────
    sep("COMPARAISON")
    types_per_n = {n: set(results[n]["types"].keys()) for n in SAMPLE_SIZES}
    for n in SAMPLE_SIZES:
        print(f"  n={n:<3} -> {len(types_per_n[n])} type(s) PII distinct(s), "
              f"{results[n]['elapsed_seconds']}s")

    new_types_steps: dict[str, list[str]] = {}
    print()
    for prev_n, n in zip(SAMPLE_SIZES, SAMPLE_SIZES[1:]):
        new_types = types_per_n[n] - types_per_n[prev_n]
        new_types_steps[f"{n}_vs_{prev_n}"] = sorted(new_types)
        print(f"  Nouveaux types en passant de {prev_n} -> {n} : {sorted(new_types) or '(aucun)'}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    ref_n = 50
    next_n = next((n for n in SAMPLE_SIZES if n > ref_n), None)
    sep(f"CONCLUSION — pourquoi n={ref_n} ?", char="-")
    if next_n is None:
        conclusion = f"Pas de taille superieure a {ref_n} testee, impossible de conclure sur la convergence."
    else:
        new_beyond_50 = new_types_steps[f"{next_n}_vs_{ref_n}"]
        if not new_beyond_50:
            conclusion = (
                f"n={next_n} n'apporte AUCUN type PII supplementaire par rapport a n={ref_n} sur ce corpus "
                f"({collection_name}). Les {len(types_per_n[ref_n])} types detectes a n={ref_n} couvrent deja "
                f"l'ensemble des types trouvables meme en augmentant l'echantillon. "
                f"n={ref_n} est donc un point de convergence raisonnable : au-dela, on paie plus de temps "
                f"Presidio (appel synchrone, bloquant, dans __init__ de la pipeline) sans gain de "
                f"couverture des types PII."
            )
        else:
            conclusion = (
                f"n={next_n} apporte {len(new_beyond_50)} type(s) PII supplementaire(s) par rapport a n={ref_n} "
                f"sur ce corpus ({collection_name}) : {new_beyond_50}. Cela suggere que n={ref_n} "
                f"ne capture pas encore tous les types PII presents ; un echantillon plus grand ou une "
                f"strategie de rappel differente pourrait etre justifiee pour ce corpus."
            )
    for line in [conclusion[i:i + 88] for i in range(0, len(conclusion), 88)]:
        print(f"  {line}")

    # ── Sauvegarde JSON ────────────────────────────────────────────────────────
    output = {
        "chroma_dir": chroma_dir,
        "collection_name": collection_name,
        "total_chunks_in_collection": total_chunks,
        "seed": bootstrap.seed,
        "sample_sizes": SAMPLE_SIZES,
        "results": {str(n): results[n] for n in SAMPLE_SIZES},
        "comparison": {
            "types_per_n": {str(n): sorted(types_per_n[n]) for n in SAMPLE_SIZES},
            "new_types_steps": new_types_steps,
        },
        "conclusion": conclusion,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Résultats enregistrés dans : {output_json}\n")


if __name__ == "__main__":
    main()
