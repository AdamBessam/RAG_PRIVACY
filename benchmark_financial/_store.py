"""
ChromaStore isolé pour le benchmark financier.
Utilise le répertoire benchmark_financial/chroma_db/ au lieu du chroma_db racine du projet.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from embeddings.embedder import Embedder


class FinancialStore:
    """
    Wrapper ChromaDB minimal, compatible avec l'interface NaiveRAG,
    mais pointant sur benchmark_financial/chroma_db/.
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
        print(f"Collection '{collection_name}' prete ({self.collection.count()} chunks indexes)")

    def index_chunks(self, chunks: list[dict], batch_size: int = 100):
        if self.collection.count() > 0:
            print(f"Collection deja indexee ({self.collection.count()} chunks) — skip")
            return

        print(f"Indexation de {len(chunks)} chunks...")
        for i in tqdm(range(0, len(chunks), batch_size), desc="ChromaDB"):
            batch = chunks[i:i + batch_size]
            texts      = [c["text"]     for c in batch]
            ids        = [c["chunk_id"] for c in batch]
            embeddings = self.embedder.embed_texts(texts, batch_size=batch_size).tolist()
            metadatas  = [
                {
                    "doc_id":       c["doc_id"],
                    "char_start":   c["char_start"],
                    "char_end":     c["char_end"],
                    "n_pii":        len(c["pii_entities"]),
                    "pii_entities": json.dumps(c["pii_entities"]),
                    "company":      c.get("company", ""),
                    "row_id":       c.get("row_id", 0),
                }
                for c in batch
            ]
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        print(f"Indexation terminee : {self.collection.count()} chunks")

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
            })
            if len(chunks) >= top_k:
                break
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print("Collection reinitialisee")
