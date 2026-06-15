"""
Test du bootstrap CPB v3 sur le corpus FINANCIER (benchmark_financial/chroma_db/).

Usage:
    python -X utf8 countermeasure_v3/test_bootstrap_financial.py
"""
import io
import logging
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("cpb_bootstrap_v3")
logger.setLevel(logging.DEBUG)

# Import direct (évite __init__.py qui charge spacy)
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "cpb_bootstrap_v3",
    os.path.join(os.path.dirname(__file__), "cpb_bootstrap_v3.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CPBBootstrapV3 = _mod.CPBBootstrapV3

from benchmark_financial._store import FinancialStore
from benchmark_financial.config import CHROMA_DIR, COLLECTION_NAME


def sep(title: str, char: str = "="):
    width = 65
    print(f"\n{char * width}")
    print(f"  {title}")
    print(char * width)


def main():
    # ── 1. Connexion ChromaDB financier ──────────────────────────────
    sep("CONNEXION CHROMADB — CORPUS FINANCIER")
    store = FinancialStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    print(f"  Chunks disponibles : {store.count()}")

    collection = store.collection
    bootstrap = CPBBootstrapV3(store=collection)

    # ── 2. Etape 0a — Decouverte PII ─────────────────────────────────
    sep("ETAPE 0a — Decouverte des types PII via Presidio")
    learned_types = bootstrap._step_0a()
    print(f"\n  {len(learned_types)} types PII trouves dans le corpus :")
    for t in sorted(learned_types):
        print(f"    - {t}")

    # ── 3. Etape 0b — Inference du domaine ───────────────────────────
    sep("ETAPE 0b — Inference du domaine (Llama 3.1 8B)")
    chunks_sample = bootstrap._sample_chunks(50)
    print(f"  Chunks echantillonnes : {len(chunks_sample)}")
    if chunks_sample:
        print(f"  Extrait (premier chunk, 200 car.) :")
        print(f"    \"{chunks_sample[0][:200]}...\"")

    domain, confidence = bootstrap._step_0b(chunks_sample)
    print(f"\n  >>> Domaine infere   : {domain.upper()}")
    print(f"  >>> Confiance Llama  : {confidence:.2f}")

    # ── 4. Etape 0c — Categories + hints ─────────────────────────────
    sep("ETAPE 0c — Generation categories + hints Presidio (Llama 3.1 8B)")
    categories, hints = bootstrap._step_0c(domain)

    print(f"\n  {len(categories)} categories generees pour le domaine '{domain}' :\n")
    for cat in categories:
        h = hints.get(cat, set())
        hint_str = ", ".join(sorted(h)) if h else "(aucun hint)"
        print(f"  [{cat}]")
        print(f"      hints Presidio : {hint_str}")

    # ── 5. Etape 0d — Phrases d'ancre ────────────────────────────────
    sep("ETAPE 0d — Enrichissement des phrases d'ancre")
    taxonomy = bootstrap._step_0d(categories, domain, chunks_sample, hints)

    print()
    for cat, phrases in taxonomy.items():
        has_real = any("<" in p or "[" in p for p in phrases)
        origin = "reelles (Presidio)" if has_real else "synthetiques (Llama)"
        print(f"  [{cat}]  —  {len(phrases)} phrases ({origin})")
        for i, p in enumerate(phrases[:3], 1):
            print(f"    {i}. {p[:110]}")
        if len(phrases) > 3:
            print(f"       ... {len(phrases) - 3} phrase(s) supplementaire(s)")
        print()

    # ── 6. Etape 0e — Centroides SBERT ───────────────────────────────
    sep("ETAPE 0e — Centroides SBERT (all-MiniLM-L6-v2)")
    centroids = bootstrap._step_0e(taxonomy)
    print()
    for cat, vec in centroids.items():
        norm = float((vec ** 2).sum() ** 0.5)
        print(f"  [{cat}]  dim={vec.shape[0]}  norme L2={norm:.4f}  (doit etre ~1.0)")

    # ── 7. Resume final ───────────────────────────────────────────────
    sep("RESUME BOOTSTRAP — CORPUS FINANCIER", char="-")
    total_phrases = sum(len(v) for v in taxonomy.values())
    centroid_dim  = next(iter(centroids.values())).shape[0] if centroids else 0
    print(f"  domaine         : {domain}  (confiance {confidence:.2f})")
    print(f"  categories      : {categories}")
    print(f"  types PII       : {len(learned_types)}  ({sorted(learned_types)})")
    print(f"  phrases total   : {total_phrases}  reparties sur {len(taxonomy)} categories")
    print(f"  centroides      : {len(centroids)} vecteurs  dim={centroid_dim}")
    print(f"  used_fallback   : False\n")


if __name__ == "__main__":
    main()
