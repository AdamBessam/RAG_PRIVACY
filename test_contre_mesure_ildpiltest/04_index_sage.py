"""
Étape 4 — Indexation des documents synthétiques SAGE dans ChromaDB.

Charge synthetic_docs.json (généré par sage/run_pipeline.py),
chunk les textes synthétiques, et les indexe dans une collection Chroma séparée.

Collection : ildpil_sage_synthetic (isolée de la collection originale)

Usage :
    python test_contre_mesure_ildpiltest/04_index_sage.py
    python test_contre_mesure_ildpiltest/04_index_sage.py --reset
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from tqdm import tqdm

from test_contre_mesure_ildpiltest._store import IldpilTestStore
from test_contre_mesure_ildpiltest.config import (
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

SYNTHETIC_DOCS_FILE  = Path(__file__).parent / "synthetic_docs.json"
SAGE_CHROMA_DIR      = str(Path(__file__).parent / "chroma_db_sage")
SAGE_COLLECTION_NAME = "ildpil_sage_synthetic"


def chunk_synthetic_docs(synthetic_docs: list[dict],
                          chunk_size: int = CHUNK_SIZE,
                          overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Découpe les documents synthétiques en chunks (même logique que data/chunker.py).
    pii_entities = [] pour tous les chunks — le texte synthétique ne contient pas de vraie PII.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=False)
        nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000

    all_chunks  = []
    chunk_count = 0

    for doc in tqdm(synthetic_docs, desc="Chunking synthetic docs"):
        text = doc.get("synthetic_text", "").strip()
        if not text:
            continue

        spacy_doc = nlp(text, disable=["ner", "lemmatizer"])
        sentences = list(spacy_doc.sents)

        current_sents = []
        current_len   = 0

        for sent in sentences:
            sent_len = len(sent.text)

            if current_len + sent_len > chunk_size and current_sents:
                chunk_text = text[current_sents[0].start_char:current_sents[-1].end_char]
                all_chunks.append({
                    "chunk_id":     f"{doc['doc_id']}_sage_c{chunk_count}",
                    "doc_id":       doc["doc_id"],
                    "text":         chunk_text,
                    "char_start":   current_sents[0].start_char,
                    "char_end":     current_sents[-1].end_char,
                    "pii_entities": [],   # texte synthétique — aucune vraie PII
                    "sensitivity":  doc.get("sensitivity", "NOT_CONFIDENTIAL"),
                })
                chunk_count += 1

                # overlap
                overlap_sents = []
                overlap_len   = 0
                for s in reversed(current_sents):
                    if overlap_len + len(s.text) <= overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s.text)
                    else:
                        break
                current_sents = overlap_sents
                current_len   = overlap_len

            current_sents.append(sent)
            current_len += sent_len

        if current_sents:
            chunk_text = text[current_sents[0].start_char:current_sents[-1].end_char]
            all_chunks.append({
                "chunk_id":     f"{doc['doc_id']}_sage_c{chunk_count}",
                "doc_id":       doc["doc_id"],
                "text":         chunk_text,
                "char_start":   current_sents[0].start_char,
                "char_end":     current_sents[-1].end_char,
                "pii_entities": [],
                "sensitivity":  doc.get("sensitivity", "NOT_CONFIDENTIAL"),
            })
            chunk_count += 1

    print(f"{len(all_chunks)} chunks générés depuis {len(synthetic_docs)} documents synthétiques")
    return all_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Repart de zéro (supprime la collection SAGE)")
    args = parser.parse_args()

    if not SYNTHETIC_DOCS_FILE.exists():
        print(f"ERREUR : {SYNTHETIC_DOCS_FILE} introuvable.")
        print("Lancez d'abord : python sage/run_pipeline.py")
        sys.exit(1)

    with open(SYNTHETIC_DOCS_FILE, encoding="utf-8") as f:
        synthetic_docs = json.load(f)

    print(f"{len(synthetic_docs)} documents synthétiques chargés depuis {SYNTHETIC_DOCS_FILE}")

    store = IldpilTestStore(chroma_dir=SAGE_CHROMA_DIR, collection_name=SAGE_COLLECTION_NAME)

    if args.reset:
        store.reset()
        print("Collection SAGE réinitialisée")

    if store.count() > 0:
        print(f"Collection SAGE déjà indexée ({store.count()} chunks). Utilisez --reset pour réimporter.")
        return

    print("Chunking des documents synthétiques...")
    chunks = chunk_synthetic_docs(synthetic_docs)

    print(f"Indexation de {len(chunks)} chunks dans ChromaDB ({SAGE_CHROMA_DIR})...")
    store.index_chunks(chunks)

    print(f"\nIndexation SAGE terminée : {store.count()} chunks dans la collection '{SAGE_COLLECTION_NAME}'")
    print(f"Répertoire Chroma : {SAGE_CHROMA_DIR}")
    print(f"\nEtape suivante : python test_contre_mesure_ildpiltest/05_run_sage_benchmark.py")


if __name__ == "__main__":
    main()
