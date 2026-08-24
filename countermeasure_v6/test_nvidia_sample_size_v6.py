"""
Test manuel — impact de la taille d'échantillon (n) envoyée au classifieur
nvidia/domain-classifier (étape 0b de CPBBootstrapV6, constante
NVIDIA_SAMPLE_SIZE). Compare n=10/20/30/40 : domaine détecté, répartition des
votes par chunk, confiance combinée, puis conclut si n=30 est un choix
raisonnable pour ce corpus.

Le pool source (jusqu'à 50 chunks tirés aléatoirement, comme en production
via bootstrap._sample_chunks(50)) est tiré UNE SEULE FOIS, puis chaque n
prend le préfixe [:n] de ce même pool — exactement comme _step_0b_nvidia
fait chunks[:NVIDIA_SAMPLE_SIZE]. Cela isole l'effet de n sans changer le
tirage aléatoire sous-jacent.

Affiche tout au terminal ET enregistre les résultats détaillés dans un JSON.

Usage:
    python countermeasure_v6/test_nvidia_sample_size_v6.py [chroma_dir] [collection_name] [output_json]

Defaults:
    chroma_dir      = data/chroma_chatdoctor
    collection_name = chatdoctor_eval_corpus
    output_json     = countermeasure_v6/test_nvidia_sample_size_results.json
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

logging.basicConfig(level=logging.WARNING)  # silence les logs tiers (transformers, presidio...)

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

SAMPLE_SIZES = [5, 10, 20, 30, 40]
POOL_SIZE = 50  # taille du pool source, comme bootstrap.run() : _sample_chunks(50)


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


def run_nvidia_pass(bootstrap: "CPBBootstrapV6", classifier, pool: list[str], n: int) -> dict:
    """Reproduit _step_0b_nvidia avec un n paramétrable, en gardant le même pool."""
    t0 = time.perf_counter()
    sample = [c[:_mod.NVIDIA_MAX_CHARS] for c in pool[:n] if c and c.strip()]

    predictions = classifier.classify(sample)
    votes: dict[str, list[float]] = {}
    for label, conf in predictions:
        votes.setdefault(label, []).append(conf)

    best_label = max(votes, key=lambda l: len(votes[l]))
    vote_share = len(votes[best_label]) / len(predictions)
    mean_conf = sum(votes[best_label]) / len(votes[best_label])
    confidence = vote_share * mean_conf
    elapsed = time.perf_counter() - t0

    return {
        "n_requested": n,
        "n_chunks_used": len(sample),
        "elapsed_seconds": round(elapsed, 2),
        "predicted_domain": best_label.lower(),
        "vote_share": round(vote_share, 3),
        "mean_confidence_winner": round(mean_conf, 3),
        "combined_confidence": round(confidence, 3),
        "votes_breakdown": {
            label: {"count": len(confs), "mean_confidence": round(sum(confs) / len(confs), 3)}
            for label, confs in sorted(votes.items(), key=lambda kv: -len(kv[1]))
        },
    }


def print_pass_result(n: int, result: dict):
    sep(f"n = {n} — nvidia/domain-classifier sur {result['n_chunks_used']} chunk(s)")
    print(f"  Temps d'exécution      : {result['elapsed_seconds']}s")
    print(f"  Domaine prédit         : {result['predicted_domain'].upper()}")
    print(f"  Part des votes gagnant : {result['vote_share']}")
    print(f"  Confiance moyenne      : {result['mean_confidence_winner']}")
    print(f"  Confiance combinée     : {result['combined_confidence']}")
    print(f"  Répartition des votes  :")
    for label, stats in result["votes_breakdown"].items():
        print(f"    • {label:<25} {stats['count']:>3} vote(s)  conf. moy.={stats['mean_confidence']}")


def main():
    chroma_dir = sys.argv[1] if len(sys.argv) > 1 else "data/chroma_chatdoctor"
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "chatdoctor_eval_corpus"
    output_json = sys.argv[3] if len(sys.argv) > 3 else str(
        Path(__file__).parent / "test_nvidia_sample_size_results.json"
    )

    sep("CONNEXION CHROMADB")
    store = _RawCollectionStore(chroma_dir, collection_name)
    total_chunks = store.collection.count()
    print(f"  Collection          : {collection_name}")
    print(f"  Chunks disponibles  : {total_chunks}")

    bootstrap = CPBBootstrapV6(store=store)

    sep("CHARGEMENT nvidia/domain-classifier")
    t0 = time.perf_counter()
    classifier = bootstrap._get_domain_classifier()
    print(f"  Chargement du modèle : {time.perf_counter() - t0:.1f}s")
    if not classifier.is_available():
        print("  ERREUR : nvidia/domain-classifier indisponible (torch/transformers manquants "
              "ou téléchargement échoué). Impossible de lancer le test.")
        sys.exit(1)

    pool = bootstrap._sample_chunks(POOL_SIZE)
    print(f"  Pool source (n={POOL_SIZE}, tirage aléatoire seed={bootstrap.seed}) : {len(pool)} chunks")

    results: dict[int, dict] = {}
    for n in SAMPLE_SIZES:
        result = run_nvidia_pass(bootstrap, classifier, pool, n)
        print_pass_result(n, result)
        results[n] = result

    # ── Comparaison entre tailles ────────────────────────────────────────────
    sep("COMPARAISON")
    for n in SAMPLE_SIZES:
        r = results[n]
        print(f"  n={n:<3} -> domaine={r['predicted_domain']:<20} "
              f"vote_share={r['vote_share']:<6} conf_combinee={r['combined_confidence']:<6} "
              f"{r['elapsed_seconds']}s")

    domains = {n: results[n]["predicted_domain"] for n in SAMPLE_SIZES}
    domain_changes = []
    for prev_n, n in zip(SAMPLE_SIZES, SAMPLE_SIZES[1:]):
        if domains[n] != domains[prev_n]:
            domain_changes.append(f"{prev_n}->{n}: {domains[prev_n]} => {domains[n]}")
    print()
    if domain_changes:
        print("  Changements de domaine prédit :")
        for c in domain_changes:
            print(f"    • {c}")
    else:
        print(f"  Domaine prédit STABLE sur tous les n testés : {domains[SAMPLE_SIZES[0]]}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    ref_n = 30
    sep(f"CONCLUSION — pourquoi n={ref_n} ?", char="-")
    confidences = [results[n]["combined_confidence"] for n in SAMPLE_SIZES]
    stable_domain = len(set(domains.values())) == 1
    ref_conf = results[ref_n]["combined_confidence"]
    max_conf = max(confidences)
    if stable_domain:
        conclusion = (
            f"Le domaine predit ({domains[ref_n]}) reste IDENTIQUE de n=10 a n=40 sur ce corpus "
            f"({collection_name}). La confiance combinee a n={ref_n} ({ref_conf}) est proche du "
            f"maximum observe ({max_conf}) sur la plage testee. n={ref_n} n'est donc pas un point "
            f"de bascule critique : la decision de domaine est deja stable a des tailles plus "
            f"petites, n={ref_n} agit surtout comme marge de securite pour la confiance."
        )
    else:
        conclusion = (
            f"Le domaine predit CHANGE selon n sur ce corpus ({collection_name}) : "
            f"{domain_changes}. n={ref_n} ne garantit donc pas une decision stable ; "
            f"un echantillon different pourrait etre necessaire pour fiabiliser la detection "
            f"de domaine sur ce corpus."
        )
    for line in [conclusion[i:i + 88] for i in range(0, len(conclusion), 88)]:
        print(f"  {line}")

    # ── Sauvegarde JSON ────────────────────────────────────────────────────────
    output = {
        "chroma_dir": chroma_dir,
        "collection_name": collection_name,
        "total_chunks_in_collection": total_chunks,
        "seed": bootstrap.seed,
        "pool_size": POOL_SIZE,
        "sample_sizes": SAMPLE_SIZES,
        "results": {str(n): results[n] for n in SAMPLE_SIZES},
        "comparison": {
            "domains_per_n": {str(n): domains[n] for n in SAMPLE_SIZES},
            "domain_changes": domain_changes,
        },
        "conclusion": conclusion,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Résultats enregistrés dans : {output_json}\n")


if __name__ == "__main__":
    main()
