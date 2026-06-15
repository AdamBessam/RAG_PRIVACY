"""
Test manuel du bootstrap CPB v3 — affiche toutes les décisions de Llama
étape par étape dans le terminal.

Usage:
    python countermeasure_v3/test_bootstrap.py
"""
import io
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid emoji encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)  # silence les logs tiers
logger = logging.getLogger("cpb_bootstrap_v3")
logger.setLevel(logging.DEBUG)

from vectorstore.chroma_store import ChromaStore

# Import direct pour éviter le chargement de __init__.py (qui charge spacy/presidio)
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "cpb_bootstrap_v3",
    os.path.join(os.path.dirname(__file__), "cpb_bootstrap_v3.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CPBBootstrapV3 = _mod.CPBBootstrapV3


def sep(title: str, char: str = "="):
    width = 65
    print(f"\n{char * width}")
    print(f"  {title}")
    print(char * width)


def main():
    # ── 1. Connexion ChromaDB ────────────────────────────────────────
    sep("CONNEXION CHROMADB")
    store = ChromaStore()
    print(f"  Chunks disponibles : {store.count()}")

    # Le bootstrap appelle store.get() → on passe la collection brute ChromaDB
    collection = store.collection

    bootstrap = CPBBootstrapV3(store=collection)

    # ── 2. Étape 0a — Découverte PII ────────────────────────────────
    sep("ÉTAPE 0a — Découverte des types PII via Presidio")
    learned_types = bootstrap._step_0a()
    print(f"\n  {len(learned_types)} types PII trouvés dans le corpus :")
    for t in sorted(learned_types):
        print(f"    • {t}")

    # ── 3. Étape 0b — Inférence du domaine ──────────────────────────
    sep("ÉTAPE 0b — Inférence du domaine (Llama 3.1 8B)")
    chunks_sample = bootstrap._sample_chunks(50)
    print(f"  Chunks échantillonnés : {len(chunks_sample)}")
    print(f"  Extrait envoyé à Llama (premier chunk, 200 car.) :")
    if chunks_sample:
        print(f"    \"{chunks_sample[0][:200]}...\"")

    domain, confidence = bootstrap._step_0b(chunks_sample)
    print(f"\n  >>> Domaine inféré   : {domain.upper()}")
    print(f"  >>> Confiance Llama  : {confidence:.2f}")

    # ── 4. Étape 0c — Catégories + hints ────────────────────────────
    sep("ÉTAPE 0c — Génération catégories + hints Presidio (Llama 3.1 8B)")
    categories, hints = bootstrap._step_0c(domain)

    print(f"\n  {len(categories)} catégories générées pour le domaine '{domain}' :\n")
    for cat in categories:
        h = hints.get(cat, set())
        hint_str = ", ".join(sorted(h)) if h else "(aucun hint)"
        print(f"  [{cat}]")
        print(f"      hints Presidio : {hint_str}")

    # ── 5. Étape 0d — Phrases d'ancre ───────────────────────────────
    sep("ÉTAPE 0d — Enrichissement des phrases d'ancre")
    taxonomy = bootstrap._step_0d(categories, domain, chunks_sample, hints)

    print()
    for cat, phrases in taxonomy.items():
        origin = "réelles" if any(
            "[" in p or "<" in p for p in phrases
        ) else "synthétiques/génériques"
        print(f"  [{cat}]  —  {len(phrases)} phrases ({origin})")
        for i, p in enumerate(phrases[:3], 1):
            print(f"    {i}. {p[:110]}")
        if len(phrases) > 3:
            print(f"       ... {len(phrases) - 3} phrase(s) supplémentaire(s)")
        print()

    # ── 6. Étape 0e — Centroïdes SBERT ──────────────────────────────
    sep("ÉTAPE 0e — Centroïdes SBERT (all-MiniLM-L6-v2)")
    centroids = bootstrap._step_0e(taxonomy)
    print()
    for cat, vec in centroids.items():
        norm = float((vec ** 2).sum() ** 0.5)
        print(f"  [{cat}]  dim={vec.shape[0]}  norme L2={norm:.4f}  (doit être ~1.0)")

    # ── 7. Résumé final ──────────────────────────────────────────────
    sep("RÉSUMÉ BOOTSTRAP", char="-")
    total_phrases = sum(len(v) for v in taxonomy.values())
    centroid_dim  = next(iter(centroids.values())).shape[0] if centroids else 0
    print(f"  domaine         : {domain}  (confiance {confidence:.2f})")
    print(f"  catégories      : {categories}")
    print(f"  types PII       : {len(learned_types)}  ({sorted(learned_types)})")
    print(f"  phrases total   : {total_phrases}  réparties sur {len(taxonomy)} catégories")
    print(f"  centroïdes      : {len(centroids)} vecteurs  dim={centroid_dim}")
    print(f"  used_fallback   : False\n")


if __name__ == "__main__":
    main()
