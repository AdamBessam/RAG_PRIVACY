"""
Compare NaiveRAG (baseline brut, sans contre-mesure) vs CPB v4 (systeme complet,
B6 patche) sur des questions NORMALES (legitimes, pas des attaques) -- pour
verifier que le systeme ne degrade pas l'utilite sur le cas d'usage de base.

Affiche tout dans le terminal ET sauvegarde un JSON complet (utile car ce script
appelle aussi NaiveRAG, qui n'est jamais loggue par les autres scripts de
l'ablation -- on veut pouvoir comparer reponse originale vs reponse apres systeme
sans tout re-parcourir le terminal).

A LANCER SUR LA MACHINE QUI A LA VRAIE CHROMADB.

Usage:
    python test_contre_mesure_ildpiltest/test_normal_naive_vs_cpb.py
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_contre_mesure_ildpiltest.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from test_contre_mesure_ildpiltest._store import IldpilTestStore

SAMPLE_PATH = Path(__file__).parent / "normal_queries_sample.json"
OUTPUT_PATH = Path(__file__).parent / "normal_naive_vs_cpb_results.json"


def sep(title: str):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def main():
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        sample = json.load(f)
    print(f"{len(sample)} requetes 'normal' chargees depuis {SAMPLE_PATH}")

    from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
    from countermeasure_v4.cpb_ablation import AblationConfig
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG

    print("Connexion ChromaDB + chargement CPB v4 (bootstrap inclus, peut prendre 1-2 min)...")
    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV4(naive_rag=naive_rag, ablation=AblationConfig(name="full_pipeline"))

    results = []
    n_block = n_synthesize = n_mask = n_pass = n_other = 0

    for item in sample:
        sep(f"[{item['query_type']}] {item['global_id']}")
        print(f"Query: {item['query']}")

        try:
            naive_out = naive_rag.run(item["query"], top_k=TOP_K)
            naive_response = naive_out["response"]
        except Exception as exc:
            naive_response = f"ERROR: {exc}"

        print(f"\n--- Reponse ORIGINALE (NaiveRAG, sans contre-mesure) ---")
        print(naive_response)

        try:
            cpb_result = cpb.run(item["query"], top_k=TOP_K)
            cpb_response = cpb_result["response"]
            sad_decision = cpb_result.get("cpb_sad_decision", "?")
            sad_categories = cpb_result.get("cpb_sad_categories", [])
            guard_decision = cpb_result.get("cpb_response_guard_decision", "?")
        except Exception as exc:
            cpb_response = f"ERROR: {exc}"
            sad_decision = "error"
            sad_categories = []
            guard_decision = "error"

        print(f"\n--- Reponse APRES SYSTEME (CPB v4, B6 patche) ---")
        print(cpb_response)
        print(f"\nB6 decision: {sad_decision}   categories: {sad_categories}   B7 (response guard): {guard_decision}")

        if sad_decision == "block":
            n_block += 1
        elif sad_decision == "synthesize":
            n_synthesize += 1
        elif sad_decision == "mask":
            n_mask += 1
        elif sad_decision == "pass":
            n_pass += 1
        else:
            n_other += 1

        results.append({
            "global_id": item["global_id"],
            "query_type": item["query_type"],
            "query": item["query"],
            "naive_response": naive_response,
            "cpb_response": cpb_response,
            "cpb_sad_decision": sad_decision,
            "cpb_sad_categories": sad_categories,
            "cpb_response_guard_decision": guard_decision,
        })

    sep("RESUME")
    print(f"  Decisions B6 : block={n_block}  synthesize={n_synthesize}  mask={n_mask}  pass={n_pass}  autre={n_other}")
    print(f"  (sur {len(sample)} questions NORMALES -- aucune ne devrait etre bloquee)")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultats sauvegardes -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
