# experiments/test_graph_rag.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.llama_llm import LlamaLLM
from llms.mistral_llm import MistralLLM
from rag.graph_rag import GraphRAG
from knowledge_graph.graph_builder import GraphBuilder
from metrics.pii_leakage import compute_pii_leakage
from metrics.response_quality import compute_response_quality
from analysis.mlflow_logger import MLflowLogger
from data.query_generator import load_queries
from embeddings.embedder import Embedder


def run_test(rag, logger, llm_name: str, queries: list, embedder):
    print(f"\n{'='*60}")
    print(f"  TEST — {llm_name} × Graph RAG")
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
        print(f"   Query entities   : {result['query_entities']}")
        print(f"   Matched entities : {len(result['matched_entities'])}")
        print(f"   Candidats        : {result['n_candidates']}")
        print(f"   Fallback         : {result['used_fallback']}")
        print(f"   ── Privacy ──────────────────────────")
        print(f"   PII total        : {pii_result.n_pii_total}")
        print(f"   PII fuitées      : {pii_result.n_pii_leaked}")
        print(f"   Taux fuite       : {pii_result.leakage_rate:.4f}")
        print(f"   ── Qualité ──────────────────────────")
        print(f"   Quality score    : {quality_result.quality_score:.4f}")
        print(f"   Answer relevancy : {quality_result.answer_relevancy:.4f}")
        print(f"   BERTScore F1     : {quality_result.bert_score_f1:.4f}")
        print(f"   ROUGE-L          : {quality_result.rouge_l:.4f}")
        print(f"   Exact Match      : {quality_result.exact_match:.4f}")

        if pii_result.leaked_entities:
            print(f"   ── Entités fuitées ──────────────────")
            for ent in pii_result.leaked_entities[:3]:
                print(f"     [{ent['type']}] '{ent['text']}' — {ent['sensitivity']}")

        run_id = logger.log_run(
            llm_name=llm_name,
            rag_name="graph_rag",
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

    # --- Construire le KG si pas encore fait ---
    print("📥 Vérification du Knowledge Graph Neo4j...")
    builder = GraphBuilder()
    builder.build(force=False)
    builder.close()

    # --- Chargement ---
    print("\n📥 Chargement ChromaDB...")
    store    = ChromaStore()
    logger   = MLflowLogger()
    embedder = Embedder()

    # --- Requêtes ---
    print("📥 Chargement des requêtes...")
    queries = load_queries()
    print(f"   {len(queries)} requêtes chargées")

    # --- Test Llama ---
    print("\n📥 Chargement Llama 3.1 8B...")
    llama_llm = LlamaLLM()
    llama_rag = GraphRAG(store=store, llm=llama_llm)
    run_test(llama_rag, logger, llm_name="llama3.1:8b",
             queries=queries, embedder=embedder)
    llama_rag.close()

    # --- Test Mistral ---
    print("\n📥 Chargement Mistral 7B...")
    mistral_llm = MistralLLM()
    mistral_rag = GraphRAG(store=store, llm=mistral_llm)
    run_test(mistral_rag, logger, llm_name="mistral:7b",
             queries=queries, embedder=embedder)
    mistral_rag.close()
