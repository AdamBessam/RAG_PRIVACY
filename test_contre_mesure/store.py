try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from embeddings.embedder import Embedder

_PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
_COLLECTION_NAME = "test_contre_mesure"


class TestChromaStore:
    """
    ChromaDB dédié à l'interface de test.
    Collection et dossier séparés du benchmark principal.
    """

    def __init__(
        self,
        collection_name: str = _COLLECTION_NAME,
        persist_dir: str = _PERSIST_DIR,
    ):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = Embedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        """Supprime et recrée la collection pour repartir de zéro."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: list[dict], batch_size: int = 100):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            embeddings = self.embedder.embed_texts(texts, batch_size=batch_size).tolist()
            metadatas = [
                {
                    "doc_id": c["doc_id"],
                    "n_pii": 0,
                    "pii_entities": json.dumps([]),
                }
                for c in batch
            ]
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_single(query_text).tolist()
        n_results = min(top_k * 3, self.collection.count())

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
                "chunk_id": results["ids"][0][j],
                "text": results["documents"][0][j],
                "similarity_score": 1 - results["distances"][0][j],
                "doc_id": doc_id,
                "n_pii": 0,
                "pii_entities": json.loads(results["metadatas"][0][j]["pii_entities"]),
            })
            if len(chunks) >= top_k:
                break

        return chunks

    def count(self) -> int:
        return self.collection.count()
