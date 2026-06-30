"""
Petit test manuel — verifie si la nouvelle cascade Tier 3 de SADDetectorV4
(reformulation LLM -> masquage -> blocage, au lieu d'un blocage immediat des
qu'on touche 2 categories) fait vraiment la difference sur les requetes qui
declenchaient un blocage total ou un masquage lourd dans l'ancien
full_pipeline.

A LANCER SUR LA MACHINE QUI A LA VRAIE CHROMADB (le retrieval doit etre reel
pour que B6 voie le meme contexte que lors du run original).

Usage:
    python test_contre_mesure_ildpiltest/test_sad_v4_retest.py
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_contre_mesure_ildpiltest.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from test_contre_mesure_ildpiltest._store import IldpilTestStore

SAMPLE_PATH = Path(__file__).parent / "sad_v4_retest_sample.json"


def sep(title: str):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def main():
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        sample = json.load(f)
    print(f"{len(sample)} requetes chargees depuis {SAMPLE_PATH}")

    # CPB v4 a besoin du bootstrap (taxonomie + centroides) avant de tourner.
    from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
    from countermeasure_v4.cpb_ablation import AblationConfig
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG

    print("Connexion ChromaDB + chargement CPB v4 (bootstrap inclus, peut prendre 1-2 min)...")
    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV4(naive_rag=naive_rag, ablation=AblationConfig(name="full_pipeline"))

    n_block_before = n_block_after = 0
    n_synthesize = n_mask = n_pass = 0

    for item in sample:
        sep(f"[{item['query_type']}] {item['global_id']}")
        print(f"Query: {item['query']}")
        print(f"\n--- ANCIENNE reponse (full_pipeline, seuil bloquant=2 cat.) ---")
        print(item["old_full_pipeline_response"])

        was_blocked = item["old_full_pipeline_response"].strip().startswith(
            "This information cannot be disclosed"
        )
        n_block_before += int(was_blocked)

        try:
            result = cpb.run(item["query"], top_k=TOP_K)
            new_response = result["response"]
            sad_decision = result.get("cpb_sad_decision", "?")
            sad_categories = result.get("cpb_sad_categories", [])
        except Exception as exc:
            new_response = f"ERROR: {exc}"
            sad_decision = "error"
            sad_categories = []

        print(f"\n--- NOUVELLE reponse (B6 patche, seuil bloquant=3 cat. + synthese LLM) ---")
        print(new_response)
        print(f"\nB6 decision: {sad_decision}   categories: {sad_categories}")

        if new_response.strip().startswith("This information cannot be disclosed"):
            n_block_after += 1
        if sad_decision == "synthesize":
            n_synthesize += 1
        elif sad_decision == "mask":
            n_mask += 1
        elif sad_decision == "pass":
            n_pass += 1

    sep("RESUME")
    print(f"  Blocages totaux AVANT : {n_block_before}/{len(sample)}")
    print(f"  Blocages totaux APRES : {n_block_after}/{len(sample)}")
    print(f"  Decisions B6 (nouveau): synthesize={n_synthesize}  mask={n_mask}  pass={n_pass}  block={n_block_after}")
    print("\nA verifier manuellement : est-ce que les reponses 'synthesize' reformulent")
    print("correctement (generalisent les details sensibles) sans inventer d'info ni")
    print("laisser fuiter le detail original tel quel.")


if __name__ == "__main__":
    main()
