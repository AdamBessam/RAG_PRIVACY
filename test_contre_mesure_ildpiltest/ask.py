"""
Outil d'inspection manuelle — voir les données indexées et juger la qualité
d'une réponse à la main (hors métriques automatiques).

Trois usages :

  # 1. Voir ce qui est indexé (nombre de chunks + un échantillon brut)
  python test_contre_mesure_ildpiltest/ask.py --list 5

  # 2. Voir SEULEMENT les chunks récupérés pour une question (rapide, sans LLM)
  python test_contre_mesure_ildpiltest/ask.py --retrieval "What health issues did Mr Omojudi face?"

  # 3. Poser une question et comparer NaiveRAG (brut) vs CPB v4 (protégé)
  python test_contre_mesure_ildpiltest/ask.py "What health issues did Mr Omojudi face?"

Tourne sur la machine qui a la vraie ChromaDB indexée (index créé par 01_index.py).
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_contre_mesure_ildpiltest.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from test_contre_mesure_ildpiltest._store import IldpilTestStore


def sep(title: str = "") -> None:
    print("=" * 100)
    if title:
        print(f"  {title}")
        print("=" * 100)


def show_indexed(store: IldpilTestStore, n: int) -> None:
    """Affiche n chunks bruts déjà indexés (contenu + métadonnées)."""
    sep(f"{store.count()} chunks indexés dans '{COLLECTION_NAME}'  —  échantillon de {n}")
    got = store.collection.get(limit=n, include=["documents", "metadatas"])
    for i, (doc, meta) in enumerate(zip(got["documents"], got["metadatas"])):
        print(f"\n[{i}] doc_id={meta.get('doc_id')}  sensibilité={meta.get('sensitivity')}  n_pii={meta.get('n_pii')}")
        print(f"    {doc[:400]}{'…' if len(doc) > 400 else ''}")


def show_retrieval(store: IldpilTestStore, query: str, top_k: int) -> None:
    """Affiche les chunks récupérés pour une question (sans LLM)."""
    sep(f"Chunks récupérés (top_k={top_k}) pour : {query}")
    chunks = store.query(query, top_k=top_k)
    if not chunks:
        print("Aucun chunk (collection vide ?).")
        return
    for i, c in enumerate(chunks):
        print(f"\n[{i}] sim={c['similarity_score']:.3f}  doc_id={c['doc_id']}  sensibilité={c['sensitivity']}  n_pii={c['n_pii']}")
        print(f"    {c['text'][:400]}{'…' if len(c['text']) > 400 else ''}")


def show_answers(store: IldpilTestStore, query: str, top_k: int) -> None:
    """Compare la réponse NaiveRAG (brute) et la réponse CPB v4 (protégée)."""
    from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
    from countermeasure_v4.cpb_ablation import AblationConfig
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG

    print("Chargement LLM + CPB v4 (bootstrap inclus, peut prendre 1-2 min)...")
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV4(naive_rag=naive_rag, ablation=AblationConfig(name="full_pipeline"))

    show_retrieval(store, query, top_k)

    sep("Réponse ORIGINALE (NaiveRAG, SANS contre-mesure) — la qualité 'brute'")
    naive_out = naive_rag.run(query, top_k=top_k)
    print(naive_out["response"])

    sep("Réponse APRÈS CPB v4 (protégée) — ce que verrait l'utilisateur")
    cpb_result = cpb.run(query, top_k=top_k)
    sad = cpb_result.get("cpb_sad_result")
    print(cpb_result["response"])
    print(
        f"\nB6 decision: {getattr(sad, 'decision', '?')}   "
        f"categories: {getattr(sad, 'attribute_categories', [])}   "
        f"B7: {cpb_result.get('cpb_response_guard_decision')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecter l'index et poser une question.")
    parser.add_argument("query", nargs="?", help="La question à poser au RAG.")
    parser.add_argument("--list", type=int, metavar="N", help="Afficher N chunks déjà indexés puis quitter.")
    parser.add_argument("--retrieval", action="store_true", help="N'afficher que les chunks récupérés (pas de LLM).")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Nombre de chunks (défaut {TOP_K}).")
    args = parser.parse_args()

    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)

    if args.list is not None:
        show_indexed(store, args.list)
        return

    if not args.query:
        parser.error("donne une question, ou utilise --list N pour voir l'index.")

    if args.retrieval:
        show_retrieval(store, args.query, args.top_k)
    else:
        show_answers(store, args.query, args.top_k)


if __name__ == "__main__":
    main()
