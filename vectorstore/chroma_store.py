# vectorstore/chroma_store.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from embeddings.embedder import Embedder


class ChromaStore:
    """
    Base vectorielle ChromaDB persistante sur disque.
    Stocke les chunks avec leurs embeddings et métadonnées PII.
    """

    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = Embedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"✅ Collection '{collection_name}' prête "
              f"({self.collection.count()} chunks déjà indexés)")

    def index_chunks(self, chunks: list[dict], batch_size: int = 100):
        """
        Indexe tous les chunks dans ChromaDB.
        Si la collection est déjà remplie, skip l'indexation.
        """
        if self.collection.count() > 0:
            print(f"⚠️  Collection déjà indexée ({self.collection.count()} chunks) — skip")
            return

        print(f"📥 Indexation de {len(chunks)} chunks...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexation ChromaDB"):
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
                }
                for c in batch
            ]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        print(f"✅ Indexation terminée : {self.collection.count()} chunks dans ChromaDB")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed_single(query_text).tolist()

    # Sécuriser n_results pour ne pas dépasser la taille de la collection
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

            # Garde seulement le meilleur chunk par document
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
        """Supprime et recrée la collection (repart de zéro)."""
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("🗑️  Collection réinitialisée")


if __name__ == "__main__":
    # Initialise ChromaDB
    store = ChromaStore()

    # Si déjà indexé → on saute le chargement et chunking
    if store.count() == 0:
        print("⚠️  Collection vide, indexation nécessaire...")
        from data.loader import load_raw_documents
        from data.chunker import chunk_documents
        documents = load_raw_documents()
        chunks    = chunk_documents(documents)
        store.index_chunks(chunks)
    else:
        print(f"✅ Collection déjà prête — {store.count()} chunks")

    # Test de requête
    print(f"\n🔍 Test de requête :")
    query = "personal data protection and privacy rights"
    results = store.query(query, top_k=3)

    print(f"   Requête : '{query}'")
    print(f"   Top {len(results)} résultats :\n")
    for i, r in enumerate(results):
        print(f"   [{i+1}] chunk_id       : {r['chunk_id']}")
        print(f"        similarity     : {r['similarity_score']:.4f}")
        print(f"        n_pii          : {r['n_pii']}")
        print(f"        doc_id         : {r['doc_id']}")
        print(f"        texte (80 car.): {r['text'][:80]}...")
        print()
