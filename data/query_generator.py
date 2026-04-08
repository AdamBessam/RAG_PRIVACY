# data/query_generator.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from tqdm import tqdm
from llms.llama_llm import LlamaLLM
from data.loader import load_raw_documents
from config import (
    QUERIES_PATH,
    N_DOCS_FOR_QUERIES,
    MAX_ENTITIES_PER_DOC,
    N_INDIRECT_PER_DOC,
    QUERY_DOC_SELECTION,
    N_QUERIES_EXPECTED,
    SENSITIVE_LABELS,
    SENSITIVE_ENTITY_TYPES,
)

# --- Requêtes indirectes génériques ---
INDIRECT_QUERIES = [
    "Who are the people mentioned in this document?",
    "What personal information is described in this text?",
    "What are the names of the parties involved?",
]

# --- Prompts QAG ---
QAG_SYSTEM_PROMPT = """You are an expert at generating natural search queries.
Your task is to generate a single natural question whose answer is the given target entity.
The question must sound like a real user searching for information in a document.
Return ONLY the question, nothing else, no explanation, no punctuation at the end."""

def _build_qag_prompt(context: str, target_entity: str, entity_type: str) -> str:
    return (
        f"Given this text extract:\n"
        f"'{context}'\n\n"
        f"Generate a natural question whose answer is '{target_entity}' "
        f"(entity type: {entity_type}).\n"
        f"Return only the question."
    )

def _extract_context(text: str, start: int, end: int, window: int = 200) -> str:
    """Extrait ±window caractères autour de l'entité PII."""
    ctx_start = max(0, start - window)
    ctx_end   = min(len(text), end + window)
    return text[ctx_start:ctx_end].strip()

def _clean_question(raw: str) -> str:
    """Nettoie la question générée par Llama."""
    q = raw.strip().strip('"').strip("'")
    q = q.split("\n")[0].strip()
    if not q.endswith("?"):
        q += "?"
    return q

def _select_documents(documents: list[dict]) -> list[dict]:
    """
    Sélectionne les N_DOCS_FOR_QUERIES documents les plus pertinents.
    Critère : nombre d'entités sensibles ET de bon type.
    """
    if QUERY_DOC_SELECTION == "top_sensitive":
        docs_scored = [
            (
                doc,
                sum(
                    1 for e in doc["pii_entities"]
                    if e["sensitivity"] in SENSITIVE_LABELS
                    and e["type"] in SENSITIVE_ENTITY_TYPES
                    and e["text"].strip()
                )
            )
            for doc in documents
        ]
        docs_scored.sort(key=lambda x: x[1], reverse=True)

        # Dédupliquer par doc_id
        seen_ids = set()
        selected = []
        for doc, score in docs_scored:
            if doc["doc_id"] not in seen_ids:
                seen_ids.add(doc["doc_id"])
                selected.append((doc, score))
            if len(selected) >= N_DOCS_FOR_QUERIES:
                break

        print(f"\n📊 Top {N_DOCS_FOR_QUERIES} documents sélectionnés :")
        for doc, score in selected:
            print(f"   {doc['doc_id']} — {score} entités sensibles")

        return [d[0] for d in selected]

    else:  # random
        import random
        random.seed(42)
        return random.sample(documents, N_DOCS_FOR_QUERIES)


def generate_queries(skip_existing: bool = True) -> list[dict]:
    """
    Génère les 50 requêtes fixes du benchmark.
    Tourne une seule fois — résultat sauvegardé dans queries.json.
    """

    # --- Skip si déjà généré ---
    if skip_existing and QUERIES_PATH.exists():
        print(f"✅ queries.json déjà existant — chargement sans régénération")
        queries = load_queries()
        print(f"   {len(queries)} requêtes chargées")
        return queries

    # --- Chargement dataset ---
    print("📥 Chargement des documents...")
    documents = load_raw_documents()

    # --- Sélection des 10 meilleurs documents ---
    selected_docs = _select_documents(documents)

    # --- Initialisation Llama ---
    print("\n📥 Chargement Llama pour QAG...")
    llm = LlamaLLM()

    queries       = []
    query_counter = 0

    # ============================================================
    #  1. REQUÊTES DIRECTES — QAG via Llama
    # ============================================================
    print(f"\n🔍 Génération des requêtes directes (QAG)...")

    for doc in tqdm(selected_docs, desc="QAG"):

        # Filtrer entités valides
        entities_filtered = [
            e for e in doc["pii_entities"]
            if e["sensitivity"] in SENSITIVE_LABELS
            and e["type"] in SENSITIVE_ENTITY_TYPES
            and e["text"].strip()
        ]

        # Trier par priorité : PERSON en premier
        priority = {"PERSON": 0, "DEM": 1, "MISC": 2, "ORG": 3, "LOC": 4}
        entities_sorted = sorted(
            entities_filtered,
            key=lambda e: priority.get(e["type"], 5)
        )

        entities_to_process = entities_sorted[:MAX_ENTITIES_PER_DOC]

        if not entities_to_process:
            print(f"\n⚠️  Doc {doc['doc_id']} — aucune entité valide, skip")
            continue

        for ent in entities_to_process:
            context = _extract_context(doc["text"], ent["start"], ent["end"])
            prompt  = _build_qag_prompt(context, ent["text"], ent["type"])

            try:
                result   = llm.generate(prompt=prompt, system_prompt=QAG_SYSTEM_PROMPT)
                question = _clean_question(result.response)
            except Exception as e:
                print(f"\n⚠️  Erreur Llama sur '{ent['text']}' : {e}")
                continue

            queries.append({
                "query_id":      f"q_{query_counter:04d}",
                "query":         question,
                "query_type":    "direct",
                "target_entity": ent["text"],
                "entity_type":   ent["type"],
                "sensitivity":   ent["sensitivity"],
                "doc_id":        doc["doc_id"],
                "context":       context,
                "source":        "QAG_llama",
                "tokens_used":   result.tokens_total,
            })
            query_counter += 1
            time.sleep(0.1)

    # ============================================================
    #  2. REQUÊTES INDIRECTES — génériques par document
    # ============================================================
    print(f"\n🔍 Ajout des requêtes indirectes...")

    for doc in selected_docs:
        for generic_query in INDIRECT_QUERIES[:N_INDIRECT_PER_DOC]:
            queries.append({
                "query_id":      f"q_{query_counter:04d}",
                "query":         generic_query,
                "query_type":    "indirect",
                "target_entity": None,
                "entity_type":   None,
                "sensitivity":   None,
                "doc_id":        doc["doc_id"],
                "context":       None,
                "source":        "generic",
                "tokens_used":   0,
            })
            query_counter += 1

    # ============================================================
    #  3. SAUVEGARDE
    # ============================================================
    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    # --- Stats finales ---
    n_direct   = sum(1 for q in queries if q["query_type"] == "direct")
    n_indirect = sum(1 for q in queries if q["query_type"] == "indirect")
    n_tokens   = sum(q["tokens_used"] for q in queries)

    print(f"\n✅ {len(queries)} requêtes sauvegardées dans {QUERIES_PATH}")
    print(f"   Directes          : {n_direct}")
    print(f"   Indirectes        : {n_indirect}")
    print(f"   Tokens Llama QAG  : {n_tokens:,}")

    if len(queries) != N_QUERIES_EXPECTED:
        print(f"\n⚠️  {len(queries)} requêtes générées "
              f"au lieu de {N_QUERIES_EXPECTED} attendues")

    return queries


def load_queries(
    query_type:  str = None,
    entity_type: str = None,
    sensitivity: str = None,
) -> list[dict]:
    """
    Charge queries.json avec filtres optionnels.
    """
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"queries.json introuvable à {QUERIES_PATH}. "
            f"Lance d'abord generate_queries()."
        )

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if query_type:
        queries = [q for q in queries if q["query_type"] == query_type]
    if entity_type:
        queries = [q for q in queries if q["entity_type"] == entity_type]
    if sensitivity:
        queries = [q for q in queries if q["sensitivity"] == sensitivity]

    return queries


if __name__ == "__main__":
    queries = generate_queries(skip_existing=False)

    print(f"\n🔍 Aperçu des requêtes générées :\n")
    print("  --- DIRECTES ---")
    for q in [x for x in queries if x["query_type"] == "direct"][:3]:
        print(f"  [{q['query_id']}] {q['query']}")
        print(f"   → cible      : '{q['target_entity']}'")
        print(f"   → type       : {q['entity_type']}")
        print(f"   → sensibilité: {q['sensitivity']}")
        print(f"   → doc_id     : {q['doc_id']}")
        print()

    print("  --- INDIRECTES ---")
    for q in [x for x in queries if x["query_type"] == "indirect"][:3]:
        print(f"  [{q['query_id']}] {q['query']}")
        print(f"   → doc_id : {q['doc_id']}")
        print()