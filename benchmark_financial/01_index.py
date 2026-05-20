"""
Etape 1 — Chargement du fichier xlsx et indexation dans ChromaDB isolé.

Usage:
    python benchmark_financial/01_index.py
    python benchmark_financial/01_index.py --reset   # repart de zero
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from tqdm import tqdm

from benchmark_financial.config import (
    DATA_FILE, CHROMA_DIR, COLLECTION_NAME,
    COL_TEXT, COL_LABELS, COL_COMPANY, COL_NAME,
    COL_SSN, COL_EMAIL, COL_PHONE, COL_ADDRESS,
)
from benchmark_financial._store import FinancialStore
from data.chunker import chunk_documents


def parse_true_predictions(raw) -> list[dict]:
    """
    Convertit la colonne True Predictions en liste de dicts PII.
    Format source : [(start, end, 'entity_type'), ...]
    """
    if not raw or (isinstance(raw, float)):
        return []
    try:
        tuples = ast.literal_eval(str(raw))
        return [
            {"start": int(s), "end": int(e), "type": t.upper(), "text": ""}
            for s, e, t in tuples
        ]
    except Exception:
        return []


def load_documents(df: pd.DataFrame) -> list[dict]:
    """
    Transforme chaque ligne du DataFrame en document compatible avec chunk_documents().
    """
    documents = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chargement documents"):
        text = str(row[COL_TEXT]).strip()
        if not text or text == "nan":
            continue

        pii_entities = parse_true_predictions(row.get(COL_LABELS, ""))

        # Enrichit les entités avec le texte réel depuis les offsets
        for ent in pii_entities:
            try:
                ent["text"] = text[ent["start"]:ent["end"]]
            except Exception:
                ent["text"] = ""

        documents.append({
            "doc_id":       f"fin_{idx}",
            "row_id":       int(idx),
            "text":         text,
            "company":      str(row.get(COL_COMPANY, "")),
            "name":         str(row.get(COL_NAME, "")),
            "ssn":          str(row.get(COL_SSN, "")),
            "email":        str(row.get(COL_EMAIL, "")),
            "phone":        str(row.get(COL_PHONE, "")),
            "address":      str(row.get(COL_ADDRESS, "")),
            "pii_entities": pii_entities,
        })
    return documents


def add_metadata_to_chunks(chunks: list[dict], doc_meta: dict) -> list[dict]:
    """Propage les métadonnées du document (company, row_id) vers chaque chunk."""
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if doc_id in doc_meta:
            chunk["company"] = doc_meta[doc_id]["company"]
            chunk["row_id"]  = doc_meta[doc_id]["row_id"]
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Repart de zero (supprime la collection)")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"ERREUR : fichier introuvable — {DATA_FILE}")
        print("Placez le fichier xlsx dans benchmark_financial/data/financial_pii_data.xlsx")
        sys.exit(1)

    print(f"Lecture de {DATA_FILE}...")
    df = pd.read_excel(DATA_FILE, engine="openpyxl")
    print(f"{len(df)} lignes chargees, colonnes : {list(df.columns)}")

    store = FinancialStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)

    if args.reset:
        store.reset()
        print("Collection reinitialise")

    if store.count() > 0:
        print(f"Collection deja indexee ({store.count()} chunks). Utilisez --reset pour reimporter.")
        sys.exit(0)

    documents = load_documents(df)
    print(f"{len(documents)} documents valides charges")

    doc_meta = {d["doc_id"]: {"company": d["company"], "row_id": d["row_id"]} for d in documents}

    print("Chunking en cours (spaCy)...")
    chunks = chunk_documents(documents)
    chunks = add_metadata_to_chunks(chunks, doc_meta)

    store.index_chunks(chunks)
    print(f"\nIndexation terminee : {store.count()} chunks dans {CHROMA_DIR}")


if __name__ == "__main__":
    main()
