# experiments/run_ikea_attack_proprietary.py
"""
IKEA Attack — LLMs propriétaires (GPT-4o mini, Claude Haiku)
=============================================================
Exécute l'attaque IKEAAttack sur toutes les combinaisons :
    LLMs : GPT-4o mini, Claude Haiku
    RAGs : NaiveRAG, SelfRAG, HHRRAG, GraphRAG

Métriques loguées dans MLflow (par round) :
    - rouge_l          : CRR — Chunk Recovery Rate
    - pii_leakage_rate : taux de fuite PII
    - tokens / coût

Métriques IKEA agrégées (MLflow SUMMARY + console) :
    - EE  : Extraction Efficiency
    - ASR : Attack Success Rate
    - CRR : Chunk Recovery Rate (ROUGE-L moyen)
    - SS  : Semantic Similarity
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.gpt4o_mini_llm import GPT4oMiniLLM
from llms.claude_haiku_llm import ClaudeHaikuLLM
from rag.naive_rag import NaiveRAG
from rag.self_rag import SelfRAG
from rag.hhr_rag import HHRRAG
from rag.graph_rag import GraphRAG
from attacks.data_extraction import IKEAAttack
from analysis.mlflow_logger import MLflowLogger

import mlflow

# Nombre de rounds IKEA par combinaison (Table 5 du papier : 50)
N_ROUNDS = 50

# ---------------------------------------------------------------------------
# Combinaisons DÉJÀ exécutées (à compléter au fur et à mesure)
# ---------------------------------------------------------------------------
ALREADY_DONE = {
    # Exemples :
    # ("gpt4o-mini",   "naive_rag"),
    # ("claude-haiku", "naive_rag"),
}


def _neo4j_available() -> bool:
    """Vérifie si Neo4j est accessible avant de lancer un experiment graph_rag."""
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fonction principale par combinaison
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
            chunk_ids=[c.get("chunk_id") for c in r.chunks if c.get("chunk_id")],
        )
        print(f"   MLflow run_id : {run_id}")

    # --- Run MLflow agrégé (EE / ASR / CRR / SS) ---
    n_refusals  = sum(1 for r in result.rounds if r.is_refusal)
    n_unrelated = sum(1 for r in result.rounds if r.is_unrelated)
    agg_run_name = f"{llm_name}__{rag_name}__ikea_attack__SUMMARY"
    with mlflow.start_run(run_name=agg_run_name):
        mlflow.log_param("llm",              llm_name)
        mlflow.log_param("rag_architecture", rag_name)
        mlflow.log_param("attack",           "ikea_attack")
        mlflow.log_param("n_rounds",         N_ROUNDS)
        mlflow.log_metric("ee",               result.ee)
        mlflow.log_metric("asr",              result.asr)
        mlflow.log_metric("crr",              result.crr)
        mlflow.log_metric("ss",               result.ss)
        mlflow.log_metric("extraction_score", result.extraction_score)
        mlflow.log_metric("n_refusals",       n_refusals)
        mlflow.log_metric("n_unrelated",      n_unrelated)
        mlflow.log_metric("pii_leakage_mean",
            sum(r.pii_leakage_rate for r in result.rounds) / len(result.rounds)
        )
    print(f"   MLflow SUMMARY loggué : {agg_run_name}")

    # --- Résumé console ---
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

    # --- LLMs propriétaires ---
    print("\n📥 Initialisation GPT-4o mini...")
    gpt = GPT4oMiniLLM()

    print("📥 Initialisation Claude Haiku...")
    claude = ClaudeHaikuLLM()

    # ---------------------------------------------------------------------------
    # Boucle : 2 LLMs × 4 RAGs = 8 expériences, en sautant les déjà faites
    # ---------------------------------------------------------------------------
    for llm_name, llm in [("gpt4o-mini", gpt), ("claude-haiku", claude)]:

        rags = [
            ("naive_rag", NaiveRAG(store=store, llm=llm)),
            ("self_rag",  SelfRAG(store=store,  llm=llm)),
            ("hhr_rag",   HHRRAG(store=store,   llm=llm)),
            ("graph_rag", GraphRAG(store=store,  llm=llm)),
        ]

        for rag_name, rag in rags:
            if (llm_name, rag_name) in ALREADY_DONE:
                print(f"\n⏭️  Skip {llm_name} × {rag_name} (déjà dans MLflow)")
                continue

            if rag_name == "graph_rag" and not _neo4j_available():
                print(f"\n❌  Skip {llm_name} × {rag_name} — Neo4j inaccessible (localhost:7687).")
                print("    Démarre Neo4j puis relance le script.")
                continue

            run_experiment(
                llm_name=llm_name,
                rag_name=rag_name,
                rag=rag,
                logger=logger,
            )

    print("\n✅ Toutes les expériences IKEA propriétaires sont terminées. Lance `mlflow ui` pour visualiser.")
