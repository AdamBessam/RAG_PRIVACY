# experiments/run_dgea_prop.py
"""
DGEA Attack — LLMs propriétaires (Claude Haiku + GPT-4o Mini)
=============================================================
Exécute l'attaque DGEA (Dynamic Greedy Embedding Attack) contre un NaiveRAG
avec les deux LLMs propriétaires disponibles.

Basé sur : Cohen et al. (2024) arXiv:2409.08045

Combinaisons testées :
    - NaiveRAG × Claude Haiku
    - NaiveRAG × GPT-4o Mini
    - SelfRAG  × Claude Haiku
    - SelfRAG  × GPT-4o Mini
    - HHRRAG   × Claude Haiku
    - HHRRAG   × GPT-4o Mini
    - GraphRAG × Claude Haiku
    - GraphRAG × GPT-4o Mini

Métriques loguées dans MLflow (par round) :
    - gea_sim_achieved      : similarité cosinus atteinte par GEA
    - n_new_chunks          : nouveaux chunks extraits ce round
    - cumulative_rate       : taux d'extraction cumulatif (%)
    - n_chunks_retrieved    : chunks retournés par ChromaDB
    - tokens_prompt / tokens_completion / cost_usd

Métriques MLflow SUMMARY (fin de run) :
    - extraction_rate_final : % de chunks extraits sur la base totale
    - total_extracted_chunks
    - avg_gea_sim           : similarité GEA moyenne

Résultats affichés en console :
    - Extraction rate final par combinaison LLM
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from vectorstore.chroma_store import ChromaStore
from llms.claude_haiku_llm import ClaudeHaikuLLM
from llms.gpt4o_mini_llm import GPT4oMiniLLM
from rag.naive_rag import NaiveRAG
from rag.self_rag import SelfRAG
from rag.hhr_rag import HHRRAG
from rag.graph_rag import GraphRAG
from attacks.dgea import DGEAAttack, N_QUERIES, TOP_K_RETRIEVAL
from analysis.mlflow_logger import MLflowLogger
from metrics.pii_leakage import compute_pii_leakage


# ---------------------------------------------------------------------------
# Nombre de queries DGEA (paper: 800 — adapté pour notre setup)
# ---------------------------------------------------------------------------
N_QUERIES_EXP = N_QUERIES   # 40


# ---------------------------------------------------------------------------
# Combinaisons déjà exécutées (à skipper)
# ---------------------------------------------------------------------------
ALREADY_DONE = {
    ("claude-haiku", "naive_rag"),
    # ("claude-haiku", "self_rag"),
    # ("gpt4o-mini",   "naive_rag"),
}


# ---------------------------------------------------------------------------
# Vérification Neo4j (pour GraphRAG)
# ---------------------------------------------------------------------------

def _neo4j_available() -> bool:
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

def run_experiment(
    llm_name: str,
    rag_name: str,
    rag,
    store: ChromaStore,
    logger: MLflowLogger,
) -> None:

    print(f"\n{'='*65}")
    print(f"  DGEA Attack — {llm_name} × {rag_name}  ({N_QUERIES_EXP} queries)")
    print(f"{'='*65}")

    attack = DGEAAttack(
        rag=rag,
        store=store,
        n_queries=N_QUERIES_EXP,
        top_k=TOP_K_RETRIEVAL,
    )

    result = attack.run(verbose=True)

    # --- Logging MLflow : un run par query ---
    for r in result.rounds:
        pii = compute_pii_leakage(response=r.response, chunks=r.chunks_retrieved)
        logger.log_run(
            llm_name=llm_name,
            rag_name=rag_name,
            attack_name="dgea",
            query=r.query,
            response=r.response,
            tokens_prompt=r.tokens_prompt,
            tokens_completion=r.tokens_completion,
            cost_usd=r.cost_usd,
            pii_leakage_rate=pii.leakage_rate,
            n_chunks_retrieved=len(r.chunks_retrieved),
            gea_sim=r.gea_sim_achieved,
            n_texts_parsed=len(r.texts_parsed),
            n_new_chunks=r.n_new_chunks,
            rouge_l=r.cumulative_extraction_rate / 100.0,
        )

    # --- Logging MLflow : run SUMMARY avec métriques agrégées ---
    avg_gea_sim = (
        sum(r.gea_sim_achieved for r in result.rounds) / len(result.rounds)
        if result.rounds else 0.0
    )

    run_name = f"{llm_name}__{rag_name}__dgea__SUMMARY"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("llm",               llm_name)
        mlflow.log_param("rag_architecture",  rag_name)
        mlflow.log_param("attack",            "dgea")
        mlflow.log_param("n_queries",         N_QUERIES_EXP)
        mlflow.log_param("top_k_retrieval",   TOP_K_RETRIEVAL)
        mlflow.log_param("total_chunks_in_db", result.total_chunks_in_db)

        mlflow.log_metric("extraction_rate_final",    result.extraction_rate)
        mlflow.log_metric("total_extracted_chunks",   len(result.extracted_chunks))
        mlflow.log_metric("avg_gea_sim",              avg_gea_sim)

    # --- Résumé console ---
    print(f"\n{'─'*65}")
    print(f"  Résultat final : {llm_name} × {rag_name}")
    print(f"  Extraction rate : {result.extraction_rate:.4f}%")
    print(f"  Chunks extraits : {len(result.extracted_chunks)} / {result.total_chunks_in_db}")
    print(f"  GEA sim moyenne : {avg_gea_sim:.4f}")
    print(f"{'─'*65}\n")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("Initialisation ChromaStore...")
    store  = ChromaStore()

    print("Initialisation MLflow logger...")
    logger = MLflowLogger()

    print("Chargement des LLMs propriétaires...")
    claude = ClaudeHaikuLLM()
    gpt    = GPT4oMiniLLM()

    # Combinaisons : 2 LLMs × 4 RAGs = 8 expériences
    for llm_instance, llm_name in [(claude, "claude-haiku"), (gpt, "gpt4o-mini")]:

        rags = [
            ("naive_rag", NaiveRAG(store=store, llm=llm_instance)),
            ("self_rag",  SelfRAG(store=store,  llm=llm_instance)),
            ("hhr_rag",   HHRRAG(store=store,   llm=llm_instance)),
            ("graph_rag", GraphRAG(store=store,  llm=llm_instance)),
        ]

        for rag_name, rag in rags:

            if (llm_name, rag_name) in ALREADY_DONE:
                print(f"\n⏭️  Skip {llm_name} × {rag_name} (déjà dans MLflow)")
                continue

            if rag_name == "graph_rag" and not _neo4j_available():
                print(f"\n❌  Skip {llm_name} × {rag_name} — Neo4j inaccessible.")
                print("    Démarre Neo4j puis relance le script.")
                continue

            run_experiment(
                llm_name=llm_name,
                rag_name=rag_name,
                rag=rag,
                store=store,
                logger=logger,
            )

    print("\nToutes les expériences DGEA (propriétaires) terminées.")
