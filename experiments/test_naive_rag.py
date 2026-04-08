# experiments/test_naive_rag.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.llama_llm import LlamaLLM
from llms.mistral_llm import MistralLLM
from rag.naive_rag import NaiveRAG
from metrics.pii_leakage import compute_pii_leakage
from analysis.mlflow_logger import MLflowLogger
from data.query_generator import load_queries


def run_test(rag, logger, llm_name: str, queries: list):
    print(f"\n{'='*60}")
    print(f"  TEST — {llm_name} × Naive RAG")
    print(f"{'='*60}")

    for q in queries:
        print(f"\n🔍 [{q['query_type'].upper()}] {q['query']}")

        # Pipeline RAG
        result = rag.run(q["query"])

        # Métriques PII
        pii_result = compute_pii_leakage(
            response=result["response"],
            chunks=result["chunks"],
        )

        # Affichage
        print(f"   Réponse        : {result['response'][:150]}...")
        print(f"   Tokens total   : {result['tokens_total']}")
        print(f"   Coût USD       : ${result['cost_usd']:.6f}")
        print(f"   PII total      : {pii_result.n_pii_total}")
        print(f"   PII fuitées    : {pii_result.n_pii_leaked}")
        print(f"   Taux fuite     : {pii_result.leakage_rate:.4f}")
        print(f"   Taux sensible  : {pii_result.sensitive_rate:.4f}")

        if pii_result.leaked_entities:
            print(f"   Entités fuitées :")
            for ent in pii_result.leaked_entities[:3]:
                print(f"     [{ent['type']}] '{ent['text']}' — {ent['sensitivity']}")

        # Log MLflow
        run_id = logger.log_run(
            llm_name=llm_name,
            rag_name="naive_rag",
            attack_name="baseline",
            query=q["query"],
            response=result["response"],
            tokens_prompt=result["tokens_prompt"],
            tokens_completion=result["tokens_completion"],
            pii_leakage_rate=pii_result.leakage_rate,
            cost_usd=result["cost_usd"],
            n_chunks_retrieved=len(result["chunks"]),
            chunk_ids=[c["chunk_id"] for c in result["chunks"]],
        )
        print(f"   ✅ MLflow run_id : {run_id}")


if __name__ == "__main__":

    # Chargement
    print("📥 Chargement ChromaDB...")
    store  = ChromaStore()
    logger = MLflowLogger()

    # Chargement requêtes
    print("📥 Chargement des requêtes...")
    queries = load_queries()
    print(f"   {len(queries)} requêtes chargées")

    # Test Llama
    print("\n📥 Chargement Llama 3.1 8B...")
    llama_llm = LlamaLLM()
    llama_rag = NaiveRAG(store=store, llm=llama_llm)
    run_test(llama_rag, logger, llm_name="llama3.1:8b", queries=queries)

    # Test Mistral
    print("\n📥 Chargement Mistral 7B...")
    mistral_llm = MistralLLM()
    mistral_rag = NaiveRAG(store=store, llm=mistral_llm)
    run_test(mistral_rag, logger, llm_name="mistral:7b", queries=queries)

    # Résumé final
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ FINAL")
    print(f"{'='*60}")
    df = logger.get_all_runs()
    if not df.empty:
        print(f"\n  Runs loggués       : {len(df)}")
        print(f"\n  Tokens moyens par LLM :")
        print(df.groupby("llm")["tokens_total"].mean().to_string())
        print(f"\n  Taux de fuite moyen par LLM :")
        print(df.groupby("llm")["pii_leakage_rate"].mean().to_string())
        print(f"\n  Coût total estimé  : ${df['cost_usd'].sum():.6f}")
    print(f"\n✅ Lance 'mlflow ui' pour visualiser les résultats")