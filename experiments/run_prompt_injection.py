# experiments/run_prompt_injection.py
"""
Expérience : Prompt Injection Attack × 4 RAGs × 2 LLMs locaux
==============================================================
Exécute l'attaque PromptInjectionAttack sur toutes les combinaisons :
    LLMs  : Llama 3.1 8B, Mistral 7B
    RAGs  : NaiveRAG, SelfRAG, HHRRAG, GraphRAG

Pour chaque requête, les métriques suivantes sont loguées dans MLflow :
    - rouge_l          : ROUGE-L (régurgitation verbatim)
    - pii_leakage_rate : taux de fuite PII
    - tokens / coût
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.llama_llm import LlamaLLM
from llms.mistral_llm import MistralLLM
from rag.naive_rag import NaiveRAG
from rag.self_rag import SelfRAG
from rag.hhr_rag import HHRRAG
from rag.graph_rag import GraphRAG
from attacks.prompt_injection import PromptInjectionAttack
from analysis.mlflow_logger import MLflowLogger
from data.query_generator import load_queries


# ---------------------------------------------------------------------------
# Fonction principale par combinaison (llm_name, rag_name, rag_instance)
# ---------------------------------------------------------------------------

def run_experiment(llm_name: str, rag_name: str, rag, queries: list, logger: MLflowLogger):
    print(f"\n{'='*65}")
    print(f"  Prompt Injection — {llm_name} × {rag_name}")
    print(f"{'='*65}")

    attack = PromptInjectionAttack(rag=rag, llm=rag.llm)
    results = attack.run(queries, verbose=True)

    for result in results:
        run_id = logger.log_run(
            llm_name=llm_name,
            rag_name=rag_name,
            attack_name="prompt_injection",
            query=result.query,
            response=result.response,
            tokens_prompt=result.tokens_prompt,
            tokens_completion=result.tokens_completion,
            pii_leakage_rate=result.pii_leakage_rate,
            cost_usd=result.cost_usd,
            rouge_l=result.rouge_l,
            query_type=result.query_type,
            n_chunks_retrieved=result.n_chunks,
            chunk_ids=[c["chunk_id"] for c in result.chunks],
        )
        print(f"   MLflow run_id : {run_id}")

    # Résumé agrégé
    avg_rouge = PromptInjectionAttack.aggregate_rouge(results)
    avg_pii   = PromptInjectionAttack.aggregate_score(results)
    n_success = sum(1 for r in results if r.reconstruction_success)

    print(f"\n  Résumé {llm_name} × {rag_name} :")
    print(f"    ROUGE-L moyen       : {avg_rouge:.4f}")
    print(f"    PII leakage moyen   : {avg_pii:.4f}")
    print(f"    Reconstructions OK  : {n_success}/{len(results)}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Ressources partagées ---
    print("📥 Chargement ChromaDB...")
    store  = ChromaStore()
    logger = MLflowLogger()

    print("📥 Chargement des requêtes...")
    queries = load_queries()
    print(f"   {len(queries)} requêtes chargées")

    # --- LLMs ---
    print("\n📥 Chargement Llama 3.1 8B...")
    llama = LlamaLLM()

    print("📥 Chargement Mistral 7B...")
    mistral = MistralLLM()

    # ---------------------------------------------------------------------------
    # Boucle : 2 LLMs × 4 RAGs = 8 expériences
    # ---------------------------------------------------------------------------
    for llm_name, llm in [("llama3.1:8b", llama), ("mistral:7b", mistral)]:

        rags = [
            ("naive_rag",  NaiveRAG(store=store, llm=llm)),
            ("self_rag",   SelfRAG(store=store,  llm=llm)),
            ("hhr_rag",    HHRRAG(store=store,   llm=llm)),
            ("graph_rag",  GraphRAG(store=store,  llm=llm)),
        ]

        for rag_name, rag in rags:
            run_experiment(
                llm_name=llm_name,
                rag_name=rag_name,
                rag=rag,
                queries=queries,
                logger=logger,
            )

    print("\n✅ Toutes les expériences sont terminées. Lance `mlflow ui` pour visualiser.")
