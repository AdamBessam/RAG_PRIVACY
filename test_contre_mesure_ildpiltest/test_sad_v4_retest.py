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
    python test_contre_mesure_ildpiltest/test_sad_v4_retest.py sad_v4_retest_sample2.json
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_contre_mesure_ildpiltest.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from test_contre_mesure_ildpiltest._store import IldpilTestStore

SAMPLE_NAME = sys.argv[1] if len(sys.argv) > 1 else "sad_v4_retest_sample.json"
SAMPLE_PATH = Path(__file__).parent / SAMPLE_NAME


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
    from metrics.pii_leakage import compute_pii_leakage

    print("Connexion ChromaDB + chargement CPB v4 (bootstrap inclus, peut prendre 1-2 min)...")
    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV4(naive_rag=naive_rag, ablation=AblationConfig(name="full_pipeline"))

    n_block_before = n_block_after = 0
    n_synthesize = n_mask = n_pass = n_reask = n_other = 0
    leaked_before_total = leaked_after_total = pii_total_total = 0

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
            raw_chunks = result.get("raw_chunks", [])
            sad_decision = result.get("cpb_sad_decision", "?")
            sad_categories = result.get("cpb_sad_categories", [])
        except Exception as exc:
            new_response = f"ERROR: {exc}"
            raw_chunks = []
            sad_decision = "error"
            sad_categories = []

        print(f"\n--- NOUVELLE reponse (B6 patche, seuil bloquant=3 cat. + synthese LLM) ---")
        print(new_response)
        print(f"\nB6 decision: {sad_decision}   categories: {sad_categories}")

        # Meme requete -> meme retrieval (deterministe) -> on peut comparer la
        # fuite de l'ancienne ET de la nouvelle reponse sur les MEMES chunks.
        pii_before = compute_pii_leakage(
            response=item["old_full_pipeline_response"], chunks=raw_chunks, query=item["query"],
        )
        pii_after = compute_pii_leakage(
            response=new_response, chunks=raw_chunks, query=item["query"],
        )
        print(f"\nFuite PII AVANT : {pii_before.n_pii_leaked}/{pii_before.n_pii_total} "
              f"({pii_before.leakage_rate:.0%})  {[e['text'] for e in pii_before.leaked_entities]}")
        print(f"Fuite PII APRES : {pii_after.n_pii_leaked}/{pii_after.n_pii_total} "
              f"({pii_after.leakage_rate:.0%})  {[e['text'] for e in pii_after.leaked_entities]}")

        leaked_before_total += pii_before.n_pii_leaked
        leaked_after_total += pii_after.n_pii_leaked
        pii_total_total += pii_after.n_pii_total

        if new_response.strip().startswith("This information cannot be disclosed"):
            n_block_after += 1
        if sad_decision == "synthesize":
            n_synthesize += 1
        elif sad_decision == "mask":
            n_mask += 1
        elif sad_decision == "pass":
            n_pass += 1
        elif sad_decision == "reask":
            n_reask += 1
        else:
            n_other += 1

    sep("RESUME")
    print(f"  Blocages totaux AVANT : {n_block_before}/{len(sample)}")
    print(f"  Blocages totaux APRES : {n_block_after}/{len(sample)}")
    print(f"  Decisions B6 (nouveau): synthesize={n_synthesize}  mask={n_mask}  "
          f"reask={n_reask}  pass={n_pass}  block={n_block_after}  autre={n_other}")
    rate_before = leaked_before_total / pii_total_total if pii_total_total else 0.0
    rate_after = leaked_after_total / pii_total_total if pii_total_total else 0.0
    print(f"\n  PII leakage AVANT : {leaked_before_total}/{pii_total_total} ({rate_before:.1%})")
    print(f"  PII leakage APRES : {leaked_after_total}/{pii_total_total} ({rate_after:.1%})")
    print("\nA verifier manuellement en plus du chiffre : est-ce que les reponses 'synthesize'")
    print("reformulent correctement (generalisent les details sensibles) sans inventer d'info.")


if __name__ == "__main__":
    main()
