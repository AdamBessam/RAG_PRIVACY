# experiments/test_hhr_rag_llm_prop.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.gpt4o_mini_llm import GPT4oMiniLLM
from llms.claude_haiku_llm import ClaudeHaikuLLM
from rag.hhr_rag import HHRRAG
from metrics.pii_leakage import compute_pii_leakage
from metrics.response_quality import compute_response_quality
from analysis.mlflow_logger import MLflowLogger
from data.query_generator import load_queries
from embeddings.embedder import Embedder


def run_test(rag, logger, llm_name: str, queries: list, embedder):
    print(f"\n{'='*60}")
    print(f"  TEST — {llm_name} × HHR RAG")
    print(f"{'='*60}")

    for q in queries:
        print(f"\n🔍 [{q['query_type'].upper()}] {q['query']}")

        result = rag.run(q["query"])

        pii_result = compute_pii_leakage(
            response=result["response"],
            chunks=result["chunks"],
        )

        quality_result = compute_response_quality(
            query=q["query"],
            response=result["response"],
            chunks=result["chunks"],
            target_entity=q.get("target_entity"),
            embedder=embedder,
        )

        print(f"   Réponse          : {result['response'][:150]}...")
        print(f"   Tokens total     : {result['tokens_total']}")
        print(f"   Coût USD         : ${result['cost_usd']:.6f}")
        print(f"   Docs stage 1     : {result['n_docs_stage1']}")
        print(f"   ── Privacy ──────────────────────────")
        print(f"   PII sensibles    : {pii_result.n_pii_total}")
        print(f"   PII fuitées      : {pii_result.n_pii_leaked}")
        print(f"   Taux fuite       : {pii_result.leakage_rate:.4f}")
        print(f"   ── Qualité ──────────────────────────")
        print(f"   Quality score    : {quality_result.quality_score:.4f}")
        print(f"   Answer relevancy : {quality_result.answer_relevancy:.4f}")
        print(f"   BERTScore F1     : {quality_result.bert_score_f1:.4f}")
        print(f"   ROUGE-L          : {quality_result.rouge_l:.4f}")
        print(f"   Exact Match      : {quality_result.exact_match:.4f}")

        run_id = logger.log_run(
            llm_name=llm_name,
            rag_name="hhr_rag",
            attack_name="baseline",
            query=q["query"],
            response=result["response"],
            tokens_prompt=result["tokens_prompt"],
            tokens_completion=result["tokens_completion"],
            pii_leakage_rate=pii_result.leakage_rate,
            cost_usd=result["cost_usd"],
            n_chunks_retrieved=len(result["chunks"]),
            query_type=q["query_type"],
            chunk_ids=[c["chunk_id"] for c in result["chunks"]],
            quality_score=quality_result.quality_score,
            answer_relevancy=quality_result.answer_relevancy,
            bert_score_f1=quality_result.bert_score_f1,
            exact_match=quality_result.exact_match,
        )
        print(f"   ✅ MLflow run_id : {run_id}")


if __name__ == "__main__":

    print("📥 Chargement ChromaDB...")
    store    = ChromaStore()
    logger   = MLflowLogger()
    embedder = Embedder()

    print("📥 Chargement des requêtes...")
    queries = load_queries()
    print(f"   {len(queries)} requêtes chargées")

    # Test GPT-4o Mini
    print("\n📥 Chargement GPT-4o Mini...")
    gpt_llm = GPT4oMiniLLM()
    gpt_rag = HHRRAG(store=store, llm=gpt_llm)
    run_test(gpt_rag, logger, llm_name="gpt4o-mini",
             queries=queries, embedder=embedder)

    # Test Claude Haiku
    print("\n📥 Chargement Claude Haiku...")
    claude_llm = ClaudeHaikuLLM()
    claude_rag = HHRRAG(store=store, llm=claude_llm)
    run_test(claude_rag, logger, llm_name="claude-haiku",
             queries=queries, embedder=embedder)

    # Résumé final
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ FINAL")
    print(f"{'='*60}")
    df = logger.get_all_runs()
    if not df.empty:
        print(f"\n  Runs loggués         : {len(df)}")
        print(f"\n  Tokens moyens par LLM :")
        print(df.groupby("llm")["tokens_total"].mean().to_string())
        print(f"\n  Taux de fuite moyen par LLM :")
        print(df.groupby("llm")["pii_leakage_rate"].mean().to_string())
        print(f"\n  Coût total estimé    : ${df['cost_usd'].sum():.6f}")

    print(f"\n✅ Lance 'mlflow ui' pour visualiser les résultats")
