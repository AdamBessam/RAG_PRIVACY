# experiments/test_naive_rag_claude_only.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.claude_haiku_llm import ClaudeHaikuLLM
from rag.naive_rag import NaiveRAG
from metrics.pii_leakage import compute_pii_leakage
from metrics.response_quality import compute_response_quality
from analysis.mlflow_logger import MLflowLogger
from data.query_generator import load_queries
from embeddings.embedder import Embedder


if __name__ == "__main__":

    print("📥 Chargement ChromaDB...")
    store    = ChromaStore()
    logger   = MLflowLogger()
    embedder = Embedder()

    print("📥 Chargement des requêtes...")
    queries = load_queries()
    print(f"   {len(queries)} requêtes chargées")

    print("\n📥 Chargement Claude Haiku...")
    llm = ClaudeHaikuLLM()
    rag = NaiveRAG(store=store, llm=llm)

    print(f"\n{'='*60}")
    print(f"  TEST — claude-haiku × Naive RAG")
    print(f"{'='*60}")

    for q in queries:
        print(f"\n🔍 [{q['query_type'].upper()}] {q['query']}")

        try:
            result = rag.run(q["query"])
        except Exception as e:
            print(f"   ❌ Erreur API pour '{q['query_id']}': {e} — query ignorée")
            continue

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
        print(f"   PII fuitées      : {pii_result.n_pii_leaked} / {pii_result.n_pii_total}")
        print(f"   Taux fuite       : {pii_result.leakage_rate:.4f}")
        print(f"   Quality score    : {quality_result.quality_score:.4f}")

        logger.log_run(
            llm_name="claude-haiku",
            rag_name="naive_rag",
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

    print(f"\n{'='*60}")
    print(f"  DONE — résultats loggués dans MLflow")
    print(f"  Lance 'mlflow ui' pour visualiser")
    print(f"{'='*60}")
