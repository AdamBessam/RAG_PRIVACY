"""
index_chunks.py — Minimal, isolated ChromaDB insertion step.

Deliberately imports ONLY chromadb + numpy + json (no langchain, no datasets,
no torch/sentence-transformers). On Windows, running chromadb's Rust-backed
collection.add() in the same process as those heavier native libraries
(observed with langchain + datasets + sentence-transformers loaded together,
and with CPB v3's Llama/Presidio/scispaCy/GLiNER stack) triggers an
intermittent native access violation. Keeping this step in its own minimal
process avoids that conflict.

Resumable: only inserts chunks beyond collection.count(), so re-running this
script after a crash (or being retried automatically by the caller) picks up
where it left off.

Generic: works for any cache of {ids, texts, metadatas} + parallel embeddings,
not just the main corpus — e.g. also used to index the PI claims DB.

Usage:
  python index_chunks.py [--chroma-dir PATH] [--collection NAME]
                          [--chunks-cache PATH] [--embeddings-cache PATH]
                          [--insert-batch N]
"""
import argparse
import json
import sys
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "chatdoctor_eval"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-dir", default=str(ROOT / "data" / "chroma_chatdoctor"))
    parser.add_argument("--collection", default="chatdoctor_eval_corpus")
    parser.add_argument("--chunks-cache", default=str(DATA_DIR / "chunks_cache.json"))
    parser.add_argument("--embeddings-cache", default=str(DATA_DIR / "chunks_embeddings.npy"))
    parser.add_argument("--insert-batch", type=int, default=50)
    args = parser.parse_args()

    with open(args.chunks_cache, encoding="utf-8") as f:
        cache = json.load(f)
    ids, texts, metadatas = cache["ids"], cache["texts"], cache["metadatas"]
    embeddings = np.load(args.embeddings_cache)

    client = chromadb.PersistentClient(path=args.chroma_dir, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        collection = client.create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})

    start = collection.count()
    total = len(ids)
    if start >= total:
        print(f"Already fully indexed ({start}/{total})")
        return 0

    print(f"Resuming from {start}/{total}...")
    while start < total:
        end = min(start + args.insert_batch, total)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  inserted up to {end}/{total}")
        start = end

    print(f"Done: {collection.count()} chunks indexed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
