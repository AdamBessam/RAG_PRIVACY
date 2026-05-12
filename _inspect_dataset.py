import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import chromadb
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection(CHROMA_COLLECTION_NAME)

# Stats globales
total_chunks = col.count()
print(f"Total chunks dans ChromaDB : {total_chunks}")

# Recuperer tous les metadatas (par batch pour eviter timeout)
all_metadatas = []
batch_size = 5000
offset = 0
while True:
    results = col.get(include=['metadatas'], limit=batch_size, offset=offset)
    if not results['metadatas']:
        break
    all_metadatas.extend(results['metadatas'])
    offset += batch_size
    if len(results['metadatas']) < batch_size:
        break

# Doc IDs uniques
doc_ids = set(m['doc_id'] for m in all_metadatas)
print(f"Documents uniques dans ChromaDB : {len(doc_ids)}")
print()

# PII du doc 001-58012 uniquement - affichage detaille
print("=" * 70)
print("PII du document 001-58012 (doc utilisé pour q_0008 et q_0009)")
print("=" * 70)

entities_58012 = {}
for meta in all_metadatas:
    if meta['doc_id'] != '001-58012':
        continue
    pii = json.loads(meta.get('pii_entities', '[]'))
    for ent in pii:
        key = ent['text']
        if key not in entities_58012:
            entities_58012[key] = ent

# Grouper par sensitivity
from collections import defaultdict
by_sens = defaultdict(list)
for e in entities_58012.values():
    by_sens[e['sensitivity']].append(e)

for sens in ['POLITICS', 'HEALTH', 'ETHNIC', 'SEX', 'BELIEF', 'NOT_CONFIDENTIAL']:
    ents = by_sens.get(sens, [])
    if not ents:
        continue
    print(f"\n  [{sens}] — {len(ents)} entites")
    for e in sorted(ents, key=lambda x: x.get('type','')):
        print(f"    type={e['type']:8s}  id_type={e.get('identifier_type',''):10s}  '{e['text']}'")
