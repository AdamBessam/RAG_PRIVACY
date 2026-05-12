import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import chromadb
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection(CHROMA_COLLECTION_NAME)

results = col.get(where={'doc_id': '001-58012'}, include=['metadatas'])

all_entities = {}
for meta in results['metadatas']:
    pii = json.loads(meta.get('pii_entities', '[]'))
    for ent in pii:
        key = ent['text']
        if key not in all_entities:
            all_entities[key] = ent

print(f"Doc 001-58012 : {len(results['ids'])} chunks, {len(all_entities)} entites uniques")
print()
for e in sorted(all_entities.values(), key=lambda x: x['sensitivity']):
    print(f"  [{e['sensitivity']:15s}] type={e['type']:6s}  id_type={e.get('identifier_type',''):25s}  text={e['text']}")

# Aussi lister tous les identifier_type distincts dans tout le dataset
print()
print("=== Tous les identifier_type distincts dans le dataset ===")
all_results = col.get(include=['metadatas'], limit=100000)
id_types = set()
for meta in all_results['metadatas']:
    pii = json.loads(meta.get('pii_entities', '[]'))
    for ent in pii:
        id_types.add((ent.get('identifier_type',''), ent.get('type',''), ent.get('sensitivity','')))

for it in sorted(id_types):
    print(f"  identifier_type={it[0]:25s}  entity_type={it[1]:8s}  sensitivity={it[2]}")
