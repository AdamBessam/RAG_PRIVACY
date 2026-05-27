"""
Store ChromaDB isolé pour benchmark_naive_vs_cpb.
Réutilise la collection ildpil_test_benchmark déjà indexée.
Compatible avec NaiveRAG et CPBNaiveRAG.
"""
try:
    __import__('pysqlite3')
    import sys as _sys
    _sys.modules['sqlite3'] = _sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Windows : sqlite3 natif suffit

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings

from embeddings.embedder import Embedder


class BenchmarkStore:
    """
    Wrapper ChromaDB compatible avec NaiveRAG et CPBNaiveRAG.
    Pointe sur la collection déjà indexée dans test_contre_mesure_ildpiltest/chroma_db/.
    """

    def __init__(self, chroma_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = Embedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        n = self.collection.count()
        print(f"  Collection '{collection_name}' : {n} chunks indexés")
        if n == 0:
            print("  ⚠  Collection vide ! Lancez d'abord :")
            print("     python test_contre_mesure_ildpiltest/01_index.py")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed_single(query_text).tolist()
        n_results = min(top_k * 3, self.collection.count())
        if n_results == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        seen_doc_ids = set()
        for j in range(len(results["ids"][0])):
            doc_id = results["metadatas"][0][j]["doc_id"]
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            chunks.append({
                "chunk_id":         results["ids"][0][j],
                "text":             results["documents"][0][j],
                "similarity_score": 1 - results["distances"][0][j],
                "doc_id":           doc_id,
                "n_pii":            results["metadatas"][0][j]["n_pii"],
                "pii_entities":     json.loads(results["metadatas"][0][j]["pii_entities"]),
                "sensitivity":      results["metadatas"][0][j].get("sensitivity", "NOT_CONFIDENTIAL"),
            })
            if len(chunks) >= top_k:
                break
        return chunks

    def count(self) -> int:
        return self.collection.count()
