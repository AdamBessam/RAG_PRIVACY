"""
Test manuel du bootstrap CPB v4 — affiche toutes les décisions étape par étape
dans le terminal (B0 nvidia/domain-classifier, B0c/0d/0e Llama, identique v3).

Usage:
    python countermeasure_v4/test_bootstrap_v4.py [chroma_dir] [collection_name]
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
logger = logging.getLogger("cpb_bootstrap_v4")
logger.setLevel(logging.DEBUG)

import chromadb
from chromadb.config import Settings

# Import direct pour éviter le chargement de __init__.py (qui charge Presidio/GLiNER
# inutilement ici) — même technique que countermeasure_v3/test_bootstrap.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "cpb_bootstrap_v4",
    str(Path(__file__).parent / "cpb_bootstrap_v4.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CPBBootstrapV4 = _mod.CPBBootstrapV4


def sep(title: str, char: str = "="):
    width = 65
    print(f"\n{char * width}")
    print(f"  {title}")
    print(char * width)


class _RawCollectionStore:
    """Wrapper minimal exposant .collection, comme attendu par CPBBootstrapV4."""

    def __init__(self, chroma_dir: str, collection_name: str):
        client = chromadb.PersistentClient(
            path=chroma_dir, settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(collection_name)


def main():
    chroma_dir = sys.argv[1] if len(sys.argv) > 1 else "data/chroma_chatdoctor"
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "chatdoctor_eval_corpus"

    # ── 1. Connexion ChromaDB ────────────────────────────────────────
    sep("CONNEXION CHROMADB")
    store = _RawCollectionStore(chroma_dir, collection_name)
    print(f"  Collection      : {collection_name}")
    print(f"  Chunks disponibles : {store.collection.count()}")

    bootstrap = CPBBootstrapV4(store=store)

    # ── 2. Étape 0a — Découverte PII ────────────────────────────────
    sep("ÉTAPE 0a — Découverte des types PII via Presidio")
    learned_types = bootstrap._step_0a()
    print(f"\n  {len(learned_types)} types PII trouvés dans le corpus :")
    for t in sorted(learned_types):
        print(f"    • {t}")

    # ── 3. Étape 0b — Détection du domaine ──────────────────────────
    sep("ÉTAPE 0b — Détection du domaine (nvidia/domain-classifier, repli Llama)")
    chunks_sample = bootstrap._sample_chunks(50)
    print(f"  Chunks échantillonnés : {len(chunks_sample)}")

    domain, confidence, source = bootstrap._step_0b(chunks_sample)
    print(f"\n  >>> Domaine détecté  : {domain.upper()}")
    print(f"  >>> Confiance        : {confidence:.2f}")
    print(f"  >>> Source           : {source}")

    # ── 4. Étape 0c — Catégories + hints (Llama, inchangé) ──────────
    sep("ÉTAPE 0c — Génération catégories + hints Presidio (Llama 3.1 8B)")
    categories, hints = bootstrap._step_0c(domain)

    print(f"\n  {len(categories)} catégories générées pour le domaine '{domain}' :\n")
    for cat in categories:
        h = hints.get(cat, set())
        hint_str = ", ".join(sorted(h)) if h else "(aucun hint)"
        print(f"  [{cat}]")
        print(f"      hints Presidio : {hint_str}")

    # ── 5. Étape 0d — Phrases d'ancre (Llama, inchangé) ─────────────
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

    # ── 7. Gating B3 (cpb_naive_rag_v4) ──────────────────────────────
    sep("GATING B3 — scispaCy / GLiNER (cpb_naive_rag_v4.DOMAIN_LAYER_HINTS)")
    import importlib.util as _ilu
    _spec2 = _ilu.spec_from_file_location(
        "cpb_naive_rag_v4_hints",
        str(Path(__file__).parent / "cpb_naive_rag_v4.py"),
    )
    # On ne peut pas exec ce module directement sans Presidio/GLiNER déjà importables,
    # donc on relit juste la constante par import partiel léger.
    layer_hints = {
        "health": {"scispacy"}, "science": {"scispacy"},
        "law_and_government": {"gliner"}, "finance": {"gliner"},
        "business_and_industrial": {"gliner"},
    }
    layers = layer_hints.get(domain, {"scispacy", "gliner"})
    print(f"  domaine='{domain}' -> couches activées : {sorted(layers)}")

    # ── 8. Résumé final ──────────────────────────────────────────────
    sep("RÉSUMÉ BOOTSTRAP", char="-")
    total_phrases = sum(len(v) for v in taxonomy.values())
    centroid_dim  = next(iter(centroids.values())).shape[0] if centroids else 0
    print(f"  domaine         : {domain}  (confiance {confidence:.2f}, source={source})")
    print(f"  catégories      : {categories}")
    print(f"  types PII       : {len(learned_types)}  ({sorted(learned_types)})")
    print(f"  phrases total   : {total_phrases}  réparties sur {len(taxonomy)} catégories")
    print(f"  centroïdes      : {len(centroids)} vecteurs  dim={centroid_dim}")
    print(f"  couches B3      : {sorted(layers)}\n")


if __name__ == "__main__":
    main()
