"""
Étape 5a — Annotation PII ground truth pour SAGE.

Pour chaque document synthétique (synthetic_docs.json), charge le document original
depuis HuggingFace et identifie les vraies PII qui ont survécu dans le texte synthétique.

Sortie : doc_pii_surviving.json
  {
    "ildpil_test_00000": ["Beck", "Norway", ...],  // PII originales présentes dans le synthétique
    "ildpil_test_00001": [],                        // tout anonymisé
    ...
  }

Usage :
    python test_contre_mesure_ildpiltest/05a_annotate_pii_gt.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tqdm import tqdm

from test_contre_mesure_ildpiltest.config import (
    DATASET_NAME, DATASET_SPLIT,
    SENSITIVE_ENTITY_TYPES,
)

SYNTHETIC_DOCS_FILE = Path(__file__).parent / "synthetic_docs.json"
OUTPUT_FILE         = Path(__file__).parent / "doc_pii_surviving.json"


def load_original_pii(dataset) -> dict[str, list[str]]:
    """
    Charge les PII annotées de chaque document original depuis HuggingFace.
    Retourne {doc_id: [pii_text, ...]}
    """
    pii_map = {}
    for i, sample in enumerate(tqdm(dataset, desc="Chargement PII originales")):
        doc_id = f"ildpil_test_{i:05d}"
        entity_mentions = sample.get("entity_mentions", []) or []
        pii_texts = []
        for ent in entity_mentions:
            if ent.get("entity_type", "") not in SENSITIVE_ENTITY_TYPES:
                continue
            text = ent.get("span_text", "").strip()
            if text and len(text) > 1:
                pii_texts.append(text)
        pii_map[doc_id] = pii_texts
    return pii_map


def find_surviving_pii(pii_texts: list[str], synthetic_text: str) -> list[str]:
    """
    Retourne les PII originales qui apparaissent encore dans le texte synthétique.
    Comparaison insensible à la casse.
    """
    synthetic_lower = synthetic_text.lower()
    return [p for p in pii_texts if p.lower() in synthetic_lower]


def main():
    if not SYNTHETIC_DOCS_FILE.exists():
        print(f"ERREUR : {SYNTHETIC_DOCS_FILE} introuvable.")
        print("Lancez d'abord : python sage/run_pipeline.py")
        sys.exit(1)

    with open(SYNTHETIC_DOCS_FILE, encoding="utf-8") as f:
        synthetic_docs = json.load(f)
    print(f"{len(synthetic_docs)} documents synthétiques chargés")

    print(f"\nChargement dataset original : {DATASET_NAME} (split={DATASET_SPLIT})...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    print("\nExtraction des PII originales...")
    pii_map = load_original_pii(dataset)

    print("\nAnnotation : PII survivantes dans les textes synthétiques...")
    surviving = {}
    total_pii      = 0
    total_survived = 0

    for doc in tqdm(synthetic_docs, desc="Annotation"):
        doc_id         = doc["doc_id"]
        synthetic_text = doc.get("synthetic_text", "")
        orig_pii       = pii_map.get(doc_id, [])

        survived = find_surviving_pii(orig_pii, synthetic_text)
        surviving[doc_id] = survived

        total_pii      += len(orig_pii)
        total_survived += len(survived)

    # Sauvegarde
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(surviving, f, ensure_ascii=False, indent=2)

    # Résumé
    docs_with_leak  = sum(1 for v in surviving.values() if v)
    docs_clean      = len(surviving) - docs_with_leak
    survival_rate   = total_survived / total_pii if total_pii > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  RÉSULTATS ANNOTATION PII SAGE")
    print(f"{'='*55}")
    print(f"  Documents annotés          : {len(surviving)}")
    print(f"  Total PII originales       : {total_pii}")
    print(f"  PII survivantes (dans synt): {total_survived}")
    print(f"  Taux survie PII            : {survival_rate:.1%}")
    print(f"  Docs avec PII survivantes  : {docs_with_leak} ({docs_with_leak/len(surviving):.1%})")
    print(f"  Docs entièrement anonymisés: {docs_clean} ({docs_clean/len(surviving):.1%})")
    print(f"{'='*55}")
    print(f"\n  Sortie : {OUTPUT_FILE}")
    print(f"\n  Etape suivante : python test_contre_mesure_ildpiltest/05_run_sage_benchmark.py")


if __name__ == "__main__":
    main()
