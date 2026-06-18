"""
export_collection.py — exporte une collection ChromaDB locale vers un fichier .npz
portable, pour transfert vers une autre machine (ex: serveur distant -> poste local).

À lancer SUR LA MACHINE DISTANTE (là où la collection est déjà indexée) :
  python sage_reference/export_collection.py healthcaremagic
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from vectorstore.chroma_store import ChromaStore


def main():
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "healthcaremagic"
    out_path = ROOT / f"{collection_name}_export.npz"

    store = ChromaStore(collection_name=collection_name)
    count = store.count()
    if count == 0:
        print(f"Collection '{collection_name}' vide — rien à exporter.")
        return

    print(f"Export de {count} chunks depuis '{collection_name}'...")
    result = store.collection.get(include=["embeddings", "documents", "metadatas"])

    np.savez_compressed(
        out_path,
        ids=np.array(result["ids"], dtype=object),
        embeddings=np.array(result["embeddings"], dtype=np.float32),
        documents=np.array(result["documents"], dtype=object),
        metadatas=np.array([json.dumps(m) for m in result["metadatas"]], dtype=object),
    )
    print(f"Export écrit : {out_path} ({out_path.stat().st_size / 1e6:.1f} Mo)")
    print("-> transfère ce fichier sur ta machine locale, puis lance import_collection.py")


if __name__ == "__main__":
    main()
