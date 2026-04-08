import os
from datasets import load_dataset

print("Chargement du dataset...")
ds = load_dataset('ildpil/text-anonymization-benchmark', split='train')

print(f"\n✅ Dataset chargé: {len(ds)} exemples")
print(f"\nEmplacement du cache Hugging Face:")
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
print(f"   {cache_dir}")

print(f"\nPour voir les fichiers:")
print(f"   explorer {cache_dir}")

print(f"\nStructure du dataset:")
print(f"   Colonnes: {ds.column_names}")
print(f"\nPremier exemple:")
print(ds[0])
