# debug_dataset.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import load_raw_documents
from collections import Counter

documents = load_raw_documents()

# 1. Valeurs uniques sensitivity
sensitivity_values = Counter()
type_values        = Counter()
type_per_sensitivity = {}

for doc in documents:
    for ent in doc["pii_entities"]:
        sensitivity_values[ent["sensitivity"]] += 1
        type_values[ent["type"]] += 1
        key = ent["sensitivity"]
        if key not in type_per_sensitivity:
            type_per_sensitivity[key] = Counter()
        type_per_sensitivity[key][ent["type"]] += 1

print("=== SENSITIVITY VALUES ===")
for k, v in sensitivity_values.most_common():
    print(f"  {k:20s} : {v}")

print("\n=== ENTITY TYPES ===")
for k, v in type_values.most_common():
    print(f"  {k:20s} : {v}")

print("\n=== TYPES PAR SENSITIVITY ===")
for sens, types in type_per_sensitivity.items():
    print(f"\n  [{sens}]")
    for t, c in types.most_common():
        print(f"    {t:20s} : {c}")

print("\n=== EXEMPLE ENTITES PAR TYPE ===")
seen_types = set()
for doc in documents:
    for ent in doc["pii_entities"]:
        if ent["type"] not in seen_types:
            seen_types.add(ent["type"])
            print(f"  [{ent['type']}] '{ent['text']}' — sensitivity: {ent['sensitivity']}")