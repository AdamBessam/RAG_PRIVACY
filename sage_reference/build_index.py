"""
build_index.py — à exécuter UNE FOIS depuis la racine du projet ou sage_reference/.

Charge LinhDuong/chatdoctor-200k, construit les dialogues (input + output),
split déterministe 99% retrieval / 1% test (seed fixe),
indexe le 99% dans la collection ChromaDB "healthcaremagic" (all-MiniLM-L6-v2),
et sauvegarde le 1% test en JSON pour l'évaluation d'utilité.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from vectorstore.chroma_store import ChromaStore

COLLECTION = "healthcaremagic"
SEED       = 42
TEST_RATIO = 0.01
HERE       = Path(__file__).parent


def build_dialogue(record: dict) -> str:
    inp = (record.get("input") or "").strip()
    out = (record.get("output") or "").strip()
    parts = []
    if inp:
        parts.append(f"Patient: {inp}")
    if out:
        parts.append(f"Doctor: {out}")
    return "\n".join(parts)


def main():
    print("Chargement LinhDuong/chatdoctor-200k ...")
    ds = load_dataset("LinhDuong/chatdoctor-200k", split="train")
    print(f"Dataset chargé : {len(ds)} exemples  |  colonnes : {ds.column_names}")

    split    = ds.train_test_split(test_size=TEST_RATIO, seed=SEED)
    train_ds = split["train"]
    test_ds  = split["test"]
    print(f"Retrieval (99%) : {len(train_ds)} exemples")
    print(f"Test      ( 1%) : {len(test_ds)} exemples")

    store = ChromaStore(collection_name=COLLECTION)

    if store.count() > 0:
        print(f"Collection '{COLLECTION}' déjà indexée ({store.count()} docs) — skip.")
    else:
        chunks = []
        for i, record in enumerate(train_ds):
            text = build_dialogue(record)
            if not text.strip():
                continue
            chunks.append({
                "chunk_id":     f"hcm_{i:06d}",
                "doc_id":       f"hcm_{i:06d}",
                "text":         text,
                "char_start":   0,
                "char_end":     len(text),
                "pii_entities": [],
            })
        store.index_chunks(chunks)

    test_inputs  = [r.get("input", "") or "" for r in test_ds]
    test_outputs = [r.get("output", "") or "" for r in test_ds]

    q_path = HERE / "questions" / "healthcaremagic-utility-question.json"
    t_path = HERE / "truth"     / "healthcaremagic-utility-truth.json"
    q_path.parent.mkdir(exist_ok=True)
    t_path.parent.mkdir(exist_ok=True)

    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(test_inputs, f, ensure_ascii=False, indent=2)
    with open(t_path, "w", encoding="utf-8") as f:
        json.dump(test_outputs, f, ensure_ascii=False, indent=2)

    print(f"Test set sauvegardé : {q_path.name} ({len(test_inputs)} entrées)")
    print("Terminé.")


if __name__ == "__main__":
    main()
