"""
import_collection.py — réimporte un export .npz (produit par export_collection.py)
dans une collection ChromaDB locale, sans toucher aux autres collections existantes.

À lancer SUR LA MACHINE LOCALE, après avoir transféré le fichier .npz :
  python sage_reference/import_collection.py healthcaremagic_export.npz healthcaremagic
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vectorstore.chroma_store import ChromaStore

BATCH = 500


def main():
    if len(sys.argv) < 2:
        print("Usage: python import_collection.py <export.npz> [collection_name]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    collection_name = sys.argv[2] if len(sys.argv) > 2 else in_path.stem.replace("_export", "")

    store = ChromaStore(collection_name=collection_name)
    if store.count() > 0:
        print(f"Collection '{collection_name}' déjà peuplée ({store.count()} chunks) — abandon.")
        return

    data = np.load(in_path, allow_pickle=True)
    ids = data["ids"].tolist()
    embeddings = data["embeddings"].tolist()
    documents = data["documents"].tolist()
    metadatas = [json.loads(m) for m in data["metadatas"].tolist()]

    print(f"Import de {len(ids)} chunks vers '{collection_name}'...")
    for i in range(0, len(ids), BATCH):
        store.collection.add(
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )
        print(f"  {min(i + BATCH, len(ids))}/{len(ids)}")

    print(f"Import terminé : {store.count()} chunks dans '{collection_name}'")


if __name__ == "__main__":
    main()
