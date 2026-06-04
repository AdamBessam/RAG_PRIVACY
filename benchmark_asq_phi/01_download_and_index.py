"""
Etape 1 — Téléchargement ASQ-PHI depuis GitHub et indexation dans ChromaDB.

Télécharge data/synthetic_clinical_queries.txt, parse le format
===QUERY=== / ===PHI_TAGS===, et indexe les 1051 queries comme documents.

Usage:
    python benchmark_asq_phi/01_download_and_index.py
    python benchmark_asq_phi/01_download_and_index.py --reset
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import requests
from tqdm import tqdm

from benchmark_asq_phi.config import (
    DATA_URL, DATA_FILE, CHROMA_DIR, COLLECTION_NAME,
)
from benchmark_asq_phi._store import ASQPHIStore
from benchmark_asq_phi._utils import parse_asq_phi


def download_data() -> str:
    """Télécharge le fichier depuis GitHub si absent en local."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        print(f"Fichier deja present : {DATA_FILE}")
        return DATA_FILE.read_text(encoding="utf-8")

    print(f"Telechargement depuis {DATA_URL}...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    DATA_FILE.write_text(response.text, encoding="utf-8")
    print(f"Sauvegarde dans {DATA_FILE}")
    return response.text


def build_documents(entries: list[dict]) -> list[dict]:
    """Convertit les entries ASQ-PHI en documents pour le store."""
    documents = []
    for idx, entry in enumerate(entries):
        text = entry["query"]
        documents.append({
            "doc_id":       f"asq_{idx:04d}",
            "text":         text,
            "char_start":   0,
            "char_end":     len(text),
            "pii_entities": entry["phi_entities"],
            "has_phi":      entry["has_phi"],
            # Chaque query ASQ-PHI est courte — pas besoin de chunking
            "chunk_id":     f"asq_{idx:04d}_c0",
        })
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Repart de zero")
    args = parser.parse_args()

    raw_text = download_data()

    print("Parsing du dataset ASQ-PHI...")
    entries = parse_asq_phi(raw_text)
    print(f"  Total entrees    : {len(entries)}")
    print(f"  PHI-positives    : {sum(1 for e in entries if e['has_phi'])}")
    print(f"  Hard negatives   : {sum(1 for e in entries if not e['has_phi'])}")

    documents = build_documents(entries)

    store = ASQPHIStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)

    if args.reset:
        store.reset()
        print("Collection reinitialisee")

    if store.count() > 0:
        print(f"Collection deja indexee ({store.count()} chunks). Utilisez --reset pour reimporter.")
        sys.exit(0)

    store.index_chunks(documents)
    print(f"\nIndexation terminee : {store.count()} chunks dans {CHROMA_DIR}")

    # Stats PII dans le store
    n_phi = sum(1 for d in documents if d["has_phi"])
    n_pii = sum(len(d["pii_entities"]) for d in documents)
    print(f"  Documents avec PHI : {n_phi}/{len(documents)}")
    print(f"  Total entites PHI  : {n_pii}")


if __name__ == "__main__":
    main()
