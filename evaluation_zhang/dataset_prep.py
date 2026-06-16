"""
dataset_prep.py — Download, sample, chunk, and index the Open Australian Legal Corpus.

Outputs:
  - ChromaDB collection 'zhang_eval_corpus' in data/chroma_zhang/
  - JSON file data/zhang_eval/doc_index.json  →  {doc_id: {text, metadata}}
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import json
import random
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.embedder import Embedder

SEED = 42
N_DOCS = 300
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HF_DATASET = "umarbutler/open-australian-legal-corpus"
COLLECTION_NAME = "zhang_eval_corpus"

DATA_DIR = Path(__file__).parent.parent / "data" / "zhang_eval"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_zhang"


# ── Download & sample ──────────────────────────────────────────────────────────

def download_and_sample() -> list[dict]:
    print(f"Downloading {HF_DATASET}...")
    dataset = load_dataset(HF_DATASET, split="corpus", trust_remote_code=True)
    all_docs = [row for row in dataset if row.get("text", "").strip()]
    rng = random.Random(SEED)
    sampled = rng.sample(all_docs, min(N_DOCS, len(all_docs)))
    print(f"Sampled {len(sampled)} documents (seed={SEED})")
    return sampled


def build_doc_index(docs: list[dict]) -> dict:
    """Returns {doc_id -> {text, metadata}}."""
    index = {}
    for i, doc in enumerate(docs):
        doc_id = f"doc_{i:04d}"
        text = doc.get("text", "") or ""
        index[doc_id] = {
            "text": text,
            "metadata": {k: str(v)[:256] for k, v in doc.items() if k != "text"},
        }
    return index


# ── Chunk & index ──────────────────────────────────────────────────────────────

def chunk_and_index(doc_index: dict) -> chromadb.Collection:
    """Chunks all documents and indexes them into ChromaDB. Returns the collection."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embedder = Embedder()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        existing = client.get_collection(COLLECTION_NAME)
        if existing.count() > 0:
            print(f"Collection '{COLLECTION_NAME}' already indexed ({existing.count()} chunks) — skip")
            return existing
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas, embeddings = [], [], [], []

    for doc_id, doc_data in tqdm(doc_index.items(), desc="Chunking"):
        text = doc_data["text"]
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for j, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{j:04d}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "source_doc_id": doc_id,
                "chunk_index": j,
                # Provide doc_id alias so CPBBootstrapV3 / ChromaStoreWrapper is compatible
                "doc_id": doc_id,
                "n_pii": 0,
                "pii_entities": "[]",
            })

    print(f"Embedding {len(texts)} chunks...")
    batch_size = 128
    all_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]
        embs = embedder.embed_texts(batch, batch_size=batch_size)
        all_embeddings.extend(embs.tolist())

    print("Inserting into ChromaDB...")
    insert_batch = 100
    for start in tqdm(range(0, len(ids), insert_batch), desc="Indexing"):
        collection.add(
            ids=ids[start : start + insert_batch],
            embeddings=all_embeddings[start : start + insert_batch],
            documents=texts[start : start + insert_batch],
            metadatas=metadatas[start : start + insert_batch],
        )

    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}'")
    return collection


# ── Public entry point ─────────────────────────────────────────────────────────

def prepare_dataset() -> tuple[dict, chromadb.Collection]:
    """Full pipeline: download → sample → chunk → index. Idempotent (cached)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    doc_index_path = DATA_DIR / "doc_index.json"

    if doc_index_path.exists():
        print("Loading existing doc index...")
        with open(doc_index_path, encoding="utf-8") as f:
            doc_index = json.load(f)
        print(f"{len(doc_index)} documents loaded from cache.")
    else:
        docs = download_and_sample()
        doc_index = build_doc_index(docs)
        with open(doc_index_path, "w", encoding="utf-8") as f:
            json.dump(doc_index, f, ensure_ascii=False, indent=2)
        print(f"Saved doc index → {doc_index_path}")

    collection = chunk_and_index(doc_index)
    return doc_index, collection


if __name__ == "__main__":
    doc_index, collection = prepare_dataset()
    print(f"\nDone. {len(doc_index)} docs, {collection.count()} chunks.")
