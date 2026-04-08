# data/loader.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from config import DATASET_NAME, DATASET_SPLIT
from tqdm import tqdm


def load_raw_documents() -> list[dict]:
    """
    Charge le dataset ildpil/text-anonymization-benchmark.
    Retourne une liste de dicts normalisés avec le texte et les entités PII.
    """
    print(f"📥 Chargement du dataset : {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"✅ {len(dataset)} documents chargés")

    documents = []
    for sample in tqdm(dataset, desc="Parsing documents"):
        doc = {
            "doc_id":       sample["doc_id"],
            "text":         sample["text"],
            "task":         sample["task"],
            "applicant":    sample["meta"].get("applicant", ""),
            "year":         sample["meta"].get("year", ""),
            "countries":    sample["meta"].get("countries", ""),
            "pii_entities": _parse_entities(sample["entity_mentions"]),
        }
        documents.append(doc)

    print(f"✅ {len(documents)} documents parsés")
    return documents


def _parse_entities(entity_mentions: list) -> list[dict]:
    """
    Normalise les entités PII depuis entity_mentions.
    Ne garde que les champs utiles pour le benchmark.
    """
    entities = []
    for ent in entity_mentions:
        entities.append({
            "text":               ent.get("span_text", ""),
            "type":               ent.get("entity_type", "UNKNOWN"),
            "start":              ent.get("start_offset", 0),
            "end":                ent.get("end_offset", 0),
            "identifier_type":    ent.get("identifier_type", ""),
            # confidential_status = niveau de sensibilité pour la pondération future
            "sensitivity":        ent.get("confidential_status", "NOT_CONFIDENTIAL"),
            "entity_mention_id":  ent.get("entity_mention_id", ""),
        })
    return entities


def get_stats(documents: list[dict]) -> None:
    """Affiche des statistiques de base pour vérifier le chargement."""
    total_pii = sum(len(d["pii_entities"]) for d in documents)
    avg_len   = sum(len(d["text"]) for d in documents) / len(documents)

    # Compte par type d'entité
    type_counts = {}
    for doc in documents:
        for ent in doc["pii_entities"]:
            t = ent["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n📊 Statistiques :")
    print(f"   Documents       : {len(documents)}")
    print(f"   Total entités   : {total_pii}")
    print(f"   Moy. PII/doc    : {total_pii / len(documents):.1f}")
    print(f"   Longueur moy.   : {avg_len:.0f} caractères")
    print(f"\n   Types d'entités :")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"     {t:12s} : {count}")


if __name__ == "__main__":
    documents = load_raw_documents()
    get_stats(documents)

    # Vérifie un document manuellement
    print(f"\n🔍 Exemple — doc_id : {documents[0]['doc_id']}")
    print(f"   Texte (100 car.) : {documents[0]['text'][:100]}...")
    print(f"   Applicant        : {documents[0]['applicant']}")
    print(f"   Nb entités PII   : {len(documents[0]['pii_entities'])}")
    print(f"   3 premières PII  :")
    for ent in documents[0]["pii_entities"][:3]:
        print(f"     [{ent['type']}] '{ent['text']}' "
              f"({ent['start']}:{ent['end']}) — {ent['sensitivity']}")