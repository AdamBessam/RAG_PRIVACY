# experiments/run_ikea_attack.py
"""
Expérience : IKEA Attack (Implicit Knowledge Extraction) × 4 RAGs × 2 LLMs
============================================================================
Exécute l'attaque IKEAAttack sur toutes les combinaisons :
    LLMs  : Llama 3.1 8B, Mistral 7B
    RAGs  : NaiveRAG, SelfRAG, HHRRAG, GraphRAG

Métriques loguées dans MLflow (par round) :
    - rouge_l          : CRR — Chunk Recovery Rate (régurgitation verbatim)
    - pii_leakage_rate : taux de fuite PII
    - tokens / coût

Métriques IKEA agrégées affichées en console uniquement :
    - EE  : Extraction Efficiency
    - ASR : Attack Success Rate
    - CRR : Chunk Recovery Rate (ROUGE-L moyen)
    - SS  : Semantic Similarity
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
from attacks.data_extraction import IKEAAttack
from analysis.mlflow_logger import MLflowLogger

# Nombre de rounds IKEA par combinaison (Table 5 du papier : 50)
N_ROUNDS = 50


# ---------------------------------------------------------------------------
# Fonction principale par combinaison (llm_name, rag_name, rag_instance)
# ---------------------------------------------------------------------------

def run_experiment(llm_name: str, rag_name: str, rag, logger: MLflowLogger):
    print(f"\n{'='*65}")
    print(f"  IKEA Attack — {llm_name} × {rag_name}  ({N_ROUNDS} rounds)")
    print(f"{'='*65}")

    attack = IKEAAttack(rag=rag, llm=rag.llm)
    result = attack.run(n_rounds=N_ROUNDS, verbose=True)

    # --- Logging MLflow par round ---
    for r in result.rounds:
        run_id = logger.log_run(
            llm_name=llm_name,
            rag_name=rag_name,
            attack_name="ikea_attack",
            query=r.query,
            response=r.response,
            tokens_prompt=r.tokens_prompt,
            tokens_completion=r.tokens_completion,
            pii_leakage_rate=r.pii_leakage_rate,
            cost_usd=r.cost_usd,
            rouge_l=r.rouge_l,
            query_type=r.anchor,
            n_chunks_retrieved=r.n_chunks,
            chunk_ids=[c["chunk_id"] for c in r.chunks],
        )
        print(f"   MLflow run_id : {run_id}")

    # --- Résumé agrégé (console uniquement) ---
    n_refusals  = sum(1 for r in result.rounds if r.is_refusal)
    n_unrelated = sum(1 for r in result.rounds if r.is_unrelated)

    print(f"\n  Résumé {llm_name} × {rag_name} :")
    print(f"    EE  (Extraction Efficiency) : {result.ee:.4f}")
    print(f"    ASR (Attack Success Rate)   : {result.asr:.4f}")
    print(f"    CRR (Chunk Recovery Rate)   : {result.crr:.4f}")
    print(f"    SS  (Semantic Similarity)   : {result.ss:.4f}")
    print(f"    Score extraction            : {result.extraction_score:.4f}")
    print(f"    Refus LLM                   : {n_refusals}/{N_ROUNDS}")
    print(f"    Hors-sujet                  : {n_unrelated}/{N_ROUNDS}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Ressources partagées ---
    print("📥 Chargement ChromaDB...")
    store  = ChromaStore()
    logger = MLflowLogger()

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
                logger=logger,
            )

    print("\n✅ Toutes les expériences IKEA sont terminées. Lance `mlflow ui` pour visualiser.")
