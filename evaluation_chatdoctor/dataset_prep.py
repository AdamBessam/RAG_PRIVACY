"""
dataset_prep.py — Download, sample, chunk, and index the ChatDoctor (HealthCareMagic) corpus.

Protocole identique à evaluation_zhang/dataset_prep.py (Zhang et al.), appliqué au
dataset médical LinhDuong/chatdoctor-200k afin de garder N=300 instances.

Chaque "document" = un dialogue  "Patient: {input}\nDoctor: {output}".

Outputs:
  - ChromaDB collection 'chatdoctor_eval_corpus' in data/chroma_chatdoctor/
  - JSON file data/chatdoctor_eval/doc_index.json  →  {doc_id: {text, metadata}}
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.embedder import Embedder

SEED = 42
N_DOCS = 300
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HF_DATASET = "LinhDuong/chatdoctor-200k"
HF_SPLIT = "train"
COLLECTION_NAME = "chatdoctor_eval_corpus"

DATA_DIR = Path(__file__).parent.parent / "data" / "chatdoctor_eval"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_chatdoctor"
CHUNKS_CACHE_PATH = DATA_DIR / "chunks_cache.json"
EMBEDDINGS_CACHE_PATH = DATA_DIR / "chunks_embeddings.npy"
INDEX_CHUNKS_SCRIPT = Path(__file__).parent / "index_chunks.py"
MAX_INDEXING_ATTEMPTS = 30


# ── Dialogue builder ─────────────────────────────────────────────────────────

def build_dialogue(record: dict) -> str:
    """Compose a single dialogue text from a chatdoctor record (input + output)."""
    inp = (record.get("input") or "").strip()
    out = (record.get("output") or "").strip()
    parts = []
    if inp:
        parts.append(f"Patient: {inp}")
    if out:
        parts.append(f"Doctor: {out}")
    return "\n".join(parts)


# ── Download & sample ──────────────────────────────────────────────────────────

def download_and_sample() -> list[dict]:
    print(f"Downloading {HF_DATASET}...")
    dataset = load_dataset(HF_DATASET, split=HF_SPLIT)
    all_docs = [row for row in dataset if build_dialogue(row).strip()]
    rng = random.Random(SEED)
    sampled = rng.sample(all_docs, min(N_DOCS, len(all_docs)))
    print(f"Sampled {len(sampled)} dialogues (seed={SEED})")
    return sampled


def build_doc_index(docs: list[dict]) -> dict:
    """Returns {doc_id -> {text, metadata}}."""
    index = {}
    for i, doc in enumerate(docs):
        doc_id = f"doc_{i:04d}"
        text = build_dialogue(doc)
        index[doc_id] = {
            "text": text,
            "metadata": {k: str(v)[:256] for k, v in doc.items() if k not in ("input", "output")},
        }
    return index


# ── Chunk & embed (cached to disk, no ChromaDB touched in this process) ────────

def build_chunks_cache(doc_index: dict) -> int:
    """
    Chunks all documents and embeds them, caching the result to disk:
      chunks_cache.json       → {ids, texts, metadatas}
      chunks_embeddings.npy   → float array (N, dim)

    Deliberately does NOT touch ChromaDB here. On Windows, running chromadb's
    Rust-backed collection.add() in the same process as langchain + datasets +
    sentence-transformers (all loaded for chunking/embedding) triggers an
    intermittent native access violation. The actual insertion happens in a
    separate minimal subprocess (index_chunks.py) — see run_indexing().

    Returns the total number of chunks.
    """
    if CHUNKS_CACHE_PATH.exists() and EMBEDDINGS_CACHE_PATH.exists():
        with open(CHUNKS_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Chunks cache already built: {len(cache['ids'])} chunks")
        return len(cache["ids"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    ids, texts, metadatas = [], [], []
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
    embedder = Embedder()
    batch_size = 128
    all_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]
        embs = embedder.embed_texts(batch, batch_size=batch_size)
        all_embeddings.extend(embs.tolist())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "texts": texts, "metadatas": metadatas}, f, ensure_ascii=False)
    np.save(EMBEDDINGS_CACHE_PATH, np.array(all_embeddings, dtype=np.float32))
    print(f"Chunks cache saved → {CHUNKS_CACHE_PATH} ({len(ids)} chunks)")

    return len(ids)


# ── Index into ChromaDB (isolated subprocess, auto-retried on native crash) ────

def run_indexing(total_chunks: int) -> None:
    """
    Runs index_chunks.py in its own minimal subprocess to insert the cached
    chunks into ChromaDB. Retries automatically if the subprocess dies from
    the intermittent native access violation (see build_chunks_cache docstring) —
    each attempt resumes from collection.count(), so a crash only costs the
    in-flight batch.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_INDEXING_ATTEMPTS + 1):
        print(f"Indexing attempt {attempt}/{MAX_INDEXING_ATTEMPTS}...")
        result = subprocess.run([sys.executable, str(INDEX_CHUNKS_SCRIPT)])

        if result.returncode == 0:
            print("Indexing completed.")
            return

        print(f"  Indexing subprocess crashed (exit code {result.returncode}) — retrying...")

    raise RuntimeError(
        f"Failed to index all {total_chunks} chunks after {MAX_INDEXING_ATTEMPTS} attempts. "
        "See index_chunks.py / dataset_prep.py for the known ChromaDB Rust/Windows crash."
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def prepare_dataset() -> tuple[dict, None]:
    """Full pipeline: download → sample → chunk → embed → index. Idempotent (cached)."""
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

    total_chunks = build_chunks_cache(doc_index)
    run_indexing(total_chunks)
    return doc_index, None


if __name__ == "__main__":
    doc_index, _ = prepare_dataset()
    print(f"\nDone. {len(doc_index)} docs indexed.")
