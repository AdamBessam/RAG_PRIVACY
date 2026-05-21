"""
Étape 1 — Chargement du split TEST de ildpil/text-anonymization-benchmark et indexation ChromaDB.

Ce split (555 documents) est vierge : jamais indexé dans les autres benchmarks.

Usage:
    python test_contre_mesure_ildpiltest/01_index.py
    python test_contre_mesure_ildpiltest/01_index.py --reset
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datasets import load_dataset
from tqdm import tqdm

from test_contre_mesure_ildpiltest.config import (
    DATASET_NAME, DATASET_SPLIT,
    CHROMA_DIR, COLLECTION_NAME,
    SENSITIVE_LABELS, NOT_SENSITIVE_LABEL, SENSITIVE_ENTITY_TYPES,
)
from test_contre_mesure_ildpiltest._store import IldpilTestStore
from data.chunker import chunk_documents


def get_sensitivity(spans: list) -> str:
    """Retourne le label de sensibilité le plus critique du document."""
    if not spans:
        return NOT_SENSITIVE_LABEL
    labels = set()
    for span in spans:
        lbl = span.get("confidentiality_label", NOT_SENSITIVE_LABEL)
        if isinstance(lbl, list):
            labels.update(lbl)
        else:
            labels.add(lbl)
    for sensitive in SENSITIVE_LABELS:
        if sensitive in labels:
            return sensitive
    return NOT_SENSITIVE_LABEL


def load_raw_documents(dataset) -> list[dict]:
    """Transforme chaque sample HuggingFace en dict compatible avec chunk_documents()."""
    documents = []
    for i, sample in enumerate(tqdm(dataset, desc="Chargement documents")):
        text = sample.get("text", "").strip()
        if not text:
            continue

        # Extraction des entités PII depuis les spans annotés
        spans = sample.get("spans", []) or []
        pii_entities = []
        for span in spans:
            ent_type = span.get("entity_type", "")
            if ent_type not in SENSITIVE_ENTITY_TYPES:
                continue
            start = span.get("start_offset", 0)
            end   = span.get("end_offset", 0)
            pii_entities.append({
                "text":        text[start:end] if start < end <= len(text) else "",
                "type":        ent_type,
                "start":       start,
                "end":         end,
                "sensitivity": get_sensitivity([span]),
            })

        sensitivity = get_sensitivity(spans)

        documents.append({
            "doc_id":       f"ildpil_test_{i:05d}",
            "text":         text,
            "pii_entities": pii_entities,
            "sensitivity":  sensitivity,
            "n_pii":        len(pii_entities),
        })

    return documents


def propagate_sensitivity_to_chunks(chunks: list[dict], doc_sensitivity: dict) -> list[dict]:
    """Propage le label de sensibilité du document vers chaque chunk."""
    for chunk in chunks:
        chunk["sensitivity"] = doc_sensitivity.get(chunk["doc_id"], NOT_SENSITIVE_LABEL)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Repart de zéro (supprime la collection)")
    args = parser.parse_args()

    print(f"Chargement du dataset : {DATASET_NAME} (split={DATASET_SPLIT})...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"{len(dataset)} documents dans le split {DATASET_SPLIT}")

    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)

    if args.reset:
        store.reset()

    if store.count() > 0:
        print(f"Collection déjà indexée ({store.count()} chunks). Utilisez --reset pour réimporter.")
        return

    documents = load_raw_documents(dataset)
    print(f"{len(documents)} documents valides chargés")

    doc_sensitivity = {d["doc_id"]: d["sensitivity"] for d in documents}

    print("Chunking en cours (spaCy)...")
    chunks = chunk_documents(documents)
    chunks = propagate_sensitivity_to_chunks(chunks, doc_sensitivity)
    print(f"{len(chunks)} chunks générés")

    store.index_chunks(chunks)

    print(f"\nIndexation terminée : {store.count()} chunks dans {CHROMA_DIR}")
    print(f"Documents sensibles : {sum(1 for d in documents if d['sensitivity'] != NOT_SENSITIVE_LABEL)}/{len(documents)}")


if __name__ == "__main__":
    main()
