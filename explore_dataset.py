"""
Explorez en détail la structure du dataset text-anonymization-benchmark
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
import json

print("\n" + "="*80)
print("EXPLORATION DU DATASET: ildpil/text-anonymization-benchmark")
print("="*80)

# Charger le dataset
print("\n📥 Chargement du dataset...")
dataset = load_dataset('ildpil/text-anonymization-benchmark', split='train')

# ============================================================================
# INFO GÉNÉRALE
# ============================================================================
print(f"\n📊 INFORMATIONS GÉNÉRALES")
print(f"-" * 80)
print(f"Nombre total d'exemples: {len(dataset)}")
print(f"Colonnes du dataset: {dataset.column_names}")
print(f"Type du dataset: {type(dataset)}")

# ============================================================================
# STRUCTURE DES COLONNES
# ============================================================================
print(f"\n📋 STRUCTURE DES COLONNES")
print(f"-" * 80)

example = dataset[0]
for col_name in dataset.column_names:
    col_value = example[col_name]
    col_type = type(col_value).__name__
    
    if isinstance(col_value, str):
        content_preview = col_value[:100].replace('\n', ' ') + "..." if len(col_value) > 100 else col_value
        print(f"\n✓ {col_name}")
        print(f"  Type: {col_type}")
        print(f"  Aperçu: {content_preview}")
        print(f"  Longueur type: {len(col_value)} caractères")
    
    elif isinstance(col_value, list):
        print(f"\n✓ {col_name}")
        print(f"  Type: {col_type} (liste)")
        print(f"  Nombre d'éléments: {len(col_value)}")
        if col_value:
            print(f"  Type des éléments: {type(col_value[0]).__name__}")
            print(f"  Premier élément: {str(col_value[0])[:100]}...")
    
    elif isinstance(col_value, dict):
        print(f"\n✓ {col_name}")
        print(f"  Type: {col_type} (dictionnaire)")
        print(f"  Clés: {list(col_value.keys())}")
        for key in list(col_value.keys())[:3]:
            print(f"    - {key}: {col_value[key]}")
    
    else:
        print(f"\n✓ {col_name}")
        print(f"  Type: {col_type}")
        print(f"  Valeur: {col_value}")

# ============================================================================
# DÉTAILS: COLONNES IMPORTANTES
# ============================================================================
print(f"\n\n🔍 DÉTAILS DES COLONNES IMPORTANTES")
print(f"-" * 80)

# Colonne TEXT
print(f"\n1️⃣  COLONNE 'text'")
print(f"   Description: Texte du document juridique original")
print(f"   Type: {type(example['text']).__name__}")
print(f"   Longueur moyenne: {sum(len(d['text']) for d in dataset) / len(dataset):.0f} caractères")
print(f"   Longueur min: {min(len(d['text']) for d in dataset)} caractères")
print(f"   Longueur max: {max(len(d['text']) for d in dataset)} caractères")
print(f"   Aperçu (premiers 300 caractères):")
print(f"   {example['text'][:300]}")

# Colonne TASK
print(f"\n2️⃣  COLONNE 'task'")
print(f"   Description: Description de la tâche d'anonymisation")
print(f"   Type: {type(example['task']).__name__}")
print(f"   Contenu: {example['task']}")

# Colonne META
print(f"\n3️⃣  COLONNE 'meta'")
print(f"   Description: Métadonnées sur le document")
print(f"   Type: {type(example['meta']).__name__}")
print(f"   Clés: {list(example['meta'].keys())}")
for key, value in example['meta'].items():
    print(f"     - {key}: {value}")

# Colonne ENTITY_MENTIONS
print(f"\n4️⃣  COLONNE 'entity_mentions'")
print(f"   Description: Entités PII à anonymiser (noms, dates, lieux, etc.)")
print(f"   Type: {type(example['entity_mentions']).__name__}")
print(f"   Nombre d'entités: {len(example['entity_mentions'])}")
if example['entity_mentions']:
    print(f"\n   PREMIER EXEMPLE D'ENTITÉ:")
    entite = example['entity_mentions'][0]
    for key, value in entite.items():
        print(f"     {key}: {value}")

# ============================================================================
# STATISTIQUES SUR LES ENTITÉS
# ============================================================================
print(f"\n\n📈 STATISTIQUES SUR LES ENTITÉS (PII)")
print(f"-" * 80)

all_entity_types = {}
all_identifier_types = {}
all_confidential_statuses = {}

for doc in dataset:
    for entity in doc['entity_mentions']:
        # Compter les types d'entités
        entity_type = entity.get('entity_type', 'UNKNOWN')
        all_entity_types[entity_type] = all_entity_types.get(entity_type, 0) + 1
        
        # Compter les types d'identifiants
        id_type = entity.get('identifier_type', 'UNKNOWN')
        all_identifier_types[id_type] = all_identifier_types.get(id_type, 0) + 1
        
        # Compter les statuts de confidentialité
        conf_status = entity.get('confidential_status', 'UNKNOWN')
        all_confidential_statuses[conf_status] = all_confidential_statuses.get(conf_status, 0) + 1

print(f"\n📌 TYPES D'ENTITÉS DÉTECTÉES:")
for entity_type, count in sorted(all_entity_types.items(), key=lambda x: x[1], reverse=True):
    print(f"   {entity_type}: {count} occurrences")

print(f"\n🏷️  TYPES D'IDENTIFIANTS:")
for id_type, count in sorted(all_identifier_types.items(), key=lambda x: x[1], reverse=True):
    print(f"   {id_type}: {count} occurrences")

print(f"\n🔒 STATUTS DE CONFIDENTIALITÉ:")
for conf_status, count in sorted(all_confidential_statuses.items(), key=lambda x: x[1], reverse=True):
    print(f"   {conf_status}: {count} occurrences")

# ============================================================================
# RÉPARTITION DONNÉES
# ============================================================================
print(f"\n\n📊 RÉPARTITION DES DONNÉES")
print(f"-" * 80)

# Par type de dataset
dataset_types = {}
for doc in dataset:
    dt = doc.get('dataset_type', 'UNKNOWN')
    dataset_types[dt] = dataset_types.get(dt, 0) + 1

print(f"\nRépartition par type de dataset:")
for dt, count in dataset_types.items():
    pct = (count / len(dataset)) * 100
    print(f"   {dt}: {count} documents ({pct:.1f}%)")

# ============================================================================
# EXEMPLE COMPLET
# ============================================================================
print(f"\n\n📄 EXEMPLE COMPLET (premier document)")
print(f"-" * 80)

example_doc = dataset[0]
print(f"\nTEXTE (premiere 500 caractères):")
print(f"{example_doc['text'][:500]}")
print(f"\nTASK: {example_doc['task']}")
print(f"\nMETA: {json.dumps(example_doc['meta'], indent=2)}")
print(f"\nNOMBRE D'ENTITÉS: {len(example_doc['entity_mentions'])}")
print(f"\nPREMIERES 3 ENTITÉS:")
for i, entity in enumerate(example_doc['entity_mentions'][:3]):
    print(f"\n  Entité {i+1}:")
    print(f"    Type: {entity['entity_type']}")
    print(f"    Texte: '{entity['span_text']}'")
    print(f"    Position: {entity['start_offset']}-{entity['end_offset']}")
    print(f"    Identifiant type: {entity['identifier_type']}")
    print(f"    Confidential: {entity['confidential_status']}")

print(f"\n\n{'='*80}\n")
