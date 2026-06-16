"""
recompute_pi_naive.py — Compare PI entre Naive RAG et CPB v3.

But : vérifier si le problème de PI est un bug de calcul ou une différence d'échelle.
  - Si Naive RAG PI >> CPB v3 PI → le système protège bien (preuve que CPB réduit l'identification)
  - Si Naive RAG PI ≈ CPB v3 PI  → bug ou problème d'échelle (les deux ont le même score bas)

Usage:
    python recompute_pi_naive.py [--skip-generation]
      --skip-generation  : réutilise naive_responses.json si déjà généré
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "zhang_eval"
CHROMA_ZHANG_DIR = Path(__file__).parent.parent / "data" / "chroma_zhang"
NAIVE_RESPONSES_PATH = DATA_DIR / "naive_responses.json"
NAIVE_PI_CACHE_PATH  = DATA_DIR / "naive_pi_scores.json"


# ── Réutilise le même ZhangChromaStore que run_evaluation.py ──────────────────

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
        seen: set[str] = set()
        for j in range(len(results["ids"][0])):
            meta = results["metadatas"][0][j]
            doc_id = meta.get("source_doc_id", results["ids"][0][j])
            if doc_id in seen:
                continue
            seen.add(doc_id)
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

    def count(self) -> int:
        return self.collection.count()


def run_naive_rag(attacks: list[dict]) -> list[str]:
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG

    store = ZhangChromaStore()
    llm   = LlamaLLM()
    rag   = NaiveRAG(store=store, llm=llm)

    responses: list[str] = []
    for i, attack in enumerate(attacks):
        print(f"  Naive RAG [{i + 1}/{len(attacks)}] {attack['doc_id']}...", end="\r")
        result = rag.run(attack["query"])
        responses.append(result["response"])
    print()
    return responses


def main(skip_generation: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger les données
    with open(DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        doc_index = json.load(f)
    with open(DATA_DIR / "attack_queries.json", encoding="utf-8") as f:
        attacks = json.load(f)
    print(f"{len(doc_index)} documents, {len(attacks)} attack queries\n")

    # 2. Générer (ou charger) les réponses Naive RAG
    if skip_generation and NAIVE_RESPONSES_PATH.exists():
        print("Loading cached Naive RAG responses...")
        with open(NAIVE_RESPONSES_PATH, encoding="utf-8") as f:
            naive_responses = json.load(f)
        print(f"{len(naive_responses)} responses loaded from cache.")
    else:
        print("Running Naive RAG (llama3.1:8b)...")
        naive_responses = run_naive_rag(attacks)
        with open(NAIVE_RESPONSES_PATH, "w", encoding="utf-8") as f:
            json.dump(naive_responses, f, ensure_ascii=False, indent=2)
        print(f"Naive responses saved → {NAIVE_RESPONSES_PATH}")

    # 3. Calculer (ou charger) PI pour Naive RAG
    from metric_pi import PIMetric

    pi_metric = PIMetric()
    if NAIVE_PI_CACHE_PATH.exists():
        print("\nLoading cached Naive RAG PI scores...")
        with open(NAIVE_PI_CACHE_PATH, encoding="utf-8") as f:
            naive_pi_scores = json.load(f)
    else:
        print("\nBuilding claims DB...")
        pi_metric.build_claims_db(doc_index)
        print("Computing PI for Naive RAG responses...")
        naive_pi_scores = pi_metric.compute_pi_batch(naive_responses, attacks, verbose=True)
        with open(NAIVE_PI_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(naive_pi_scores, f, ensure_ascii=False, indent=2)
        print(f"Naive PI cache saved → {NAIVE_PI_CACHE_PATH}")

    naive_pi = PIMetric.aggregate_pi(naive_pi_scores)

    # 4. Charger PI CPB v3 si disponible
    cpb_pi_cache = DATA_DIR / "pi_scores.json"
    cpb_pi = None
    if cpb_pi_cache.exists():
        with open(cpb_pi_cache, encoding="utf-8") as f:
            cpb_pi_scores = json.load(f)
        cpb_pi = PIMetric.aggregate_pi(cpb_pi_scores)

    # 5. Résumé comparatif
    print("\n" + "=" * 55)
    print("  COMPARAISON PI — Naive RAG vs CPB v3")
    print("=" * 55)
    print(f"  Naive RAG PI : {naive_pi:.4f}")
    if cpb_pi is not None:
        ratio = naive_pi / cpb_pi if cpb_pi > 0 else float("inf")
        print(f"  CPB v3    PI : {cpb_pi:.4f}")
        print(f"  Ratio (Naive/CPB) : {ratio:.2f}x")
        print()
        if ratio > 1.5:
            print("  → CPB v3 REDUIT bien l'identification (Naive >> CPB)")
            print("    L'écart d'échelle avec Zhang est dû à la taille du dataset")
            print("    (300 docs au lieu de milliers → accumulation plus faible)")
        elif ratio < 0.8:
            print("  → Anomalie : CPB v3 PI > Naive RAG PI — bug possible ?")
            print("    Vérifier que les réponses CPB sont bien différentes des Naive")
        else:
            print("  → Naive RAG PI ≈ CPB v3 PI")
            print("    Le système ne semble pas réduire l'identification")
            print("    ou PI ne capture pas bien la différence d'anonymisation")
    else:
        print("  CPB v3    PI : (pas de cache pi_scores.json trouvé)")
    print("=" * 55)

    # 6. Stats détaillées
    n_nonzero_naive = sum(1 for s in naive_pi_scores if s > 0)
    print(f"\n  Naive RAG : {n_nonzero_naive}/{len(naive_pi_scores)} queries avec PI > 0")
    if cpb_pi is not None:
        n_nonzero_cpb = sum(1 for s in cpb_pi_scores if s > 0)
        print(f"  CPB v3    : {n_nonzero_cpb}/{len(cpb_pi_scores)} queries avec PI > 0")
        print()
        print("  Interprétation :")
        print("  PI > 0 = le vrai individu est dans le top-5 identifiés")
        print("  Si Naive a plus de PI > 0 → l'attaque réussit mieux sans protection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PI: Naive RAG vs CPB v3")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Réutilise naive_responses.json existant",
    )
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
