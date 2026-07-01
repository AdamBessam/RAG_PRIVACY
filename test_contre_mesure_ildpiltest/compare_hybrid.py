"""
Comparaison NaiveRAG (dense seul) vs HybridRAG (dense + BM25 + RRF), SANS
contre-mesure, pour isoler l'effet du retrieval sur la qualité de réponse.

Pour chaque question, affiche :
  - les doc_id récupérés par chacun (voir si le document change),
  - la réponse de chacun (voir si le "no mention of X" disparaît).

Usage :
  python test_contre_mesure_ildpiltest/compare_hybrid.py                 # 6 normales
  python test_contre_mesure_ildpiltest/compare_hybrid.py --limit 6 --type direct
  python test_contre_mesure_ildpiltest/compare_hybrid.py --query "What are the contact details for Mr Gunnar Beck as listed in the case?"
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_contre_mesure_ildpiltest.config import (
    CHROMA_DIR, COLLECTION_NAME, QUERIES_FILE, TOP_K,
)
from test_contre_mesure_ildpiltest._store import IldpilTestStore


def sep(title: str = "") -> None:
    print("=" * 100)
    if title:
        print(f"  {title}")
        print("=" * 100)


def docs_line(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        sim = c.get("similarity_score")
        tag = f"{sim:.3f}" if isinstance(sim, (int, float)) else "bm25"
        parts.append(f"{c.get('doc_id')}({tag})")
    return "  ".join(parts) if parts else "(aucun)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6, help="Nombre de questions (défaut 6).")
    parser.add_argument("--type", default="normal", help="Type de requête à échantillonner (défaut 'normal').")
    parser.add_argument("--query", default=None, help="Une question précise (ignore --limit/--type).")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    # Sélection des questions
    if args.query:
        queries = [{"query": args.query, "query_type": "custom", "global_id": "custom"}]
    else:
        if not QUERIES_FILE.exists():
            sys.exit(f"ERREUR : {QUERIES_FILE} introuvable.")
        with open(QUERIES_FILE, encoding="utf-8") as f:
            allq = json.load(f)
        pool = [q for q in allq if q.get("query_type") == args.type] or allq
        queries = pool[:args.limit]
    print(f"{len(queries)} question(s) — type='{args.type}'")

    # Imports lourds tardifs
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG
    from rag.hybrid_rag import HybridRAG

    print("Init ChromaDB + LLM + index BM25 (peut prendre 1 min)...")
    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    if store.count() == 0:
        sys.exit("ERREUR : collection vide (lancez 01_index.py).")
    llm = LlamaLLM()
    naive = NaiveRAG(store=store, llm=llm)
    hybrid = HybridRAG(store=store, llm=llm)

    for q in queries:
        query = str(q["query"])
        sep(f"[{q.get('query_type')}] {q.get('global_id', '')}")
        print(f"Q: {query}\n")

        n_chunks = naive.retrieve(query, top_k=args.top_k)
        h_chunks = hybrid.retrieve(query, top_k=args.top_k)

        print(f"NAIVE  docs : {docs_line(n_chunks)}")
        print(f"HYBRID docs : {docs_line(h_chunks)}")
        changed = [c.get("doc_id") for c in h_chunks] != [c.get("doc_id") for c in n_chunks]
        print(f"→ documents {'DIFFÉRENTS (hybrid a changé le retrieval)' if changed else 'identiques'}\n")

        print("--- Réponse NAIVE (dense) ---")
        print(naive.generate(query, n_chunks).response)
        print("\n--- Réponse HYBRID (dense + BM25) ---")
        print(hybrid.generate(query, h_chunks).response)
        print()


if __name__ == "__main__":
    main()
