# experiments/test_naive_rag_mendeley.py
#
# Test Naive RAG sur 2 documents du dataset Mendeley
# (Synthetic Dataset for PII Detection and Anonymization in Financial Documents)
# But : démontrer une bonne qualité de réponse sur un nouveau dataset
# Collection ChromaDB isolée "mendeley_test" — rien de touché dans l'existant
#
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.chunker import chunk_documents
from vectorstore.chroma_store import ChromaStore
from embeddings.embedder import Embedder
from rag.naive_rag import NaiveRAG
from llms.llama_llm import LlamaLLM
from llms.mistral_llm import MistralLLM
from metrics.response_quality import compute_response_quality
from analysis.mlflow_logger import MLflowLogger

# ============================================================
# COLLECTION ISOLÉE — ne touche pas à "rag_benchmark"
# ============================================================
MENDELEY_COLLECTION = "mendeley_test"
EXCEL_PATH = Path(__file__).parent.parent / "data" / "raw" / "Testing_Set.xlsx"

# ============================================================
# REQUÊTES MANUELLES (6 questions, 3 par document)
# target_entity = valeur exacte attendue dans la réponse
# ============================================================
QUERIES = [
    # --- Doc 0 : rapport de conformité fiscale Jackson-Guzman ---
    {
        "query":         "Which company is the subject of this tax compliance review?",
        "target_entity": "Jackson-Guzman",
        "query_type":    "factual",
    },
    {
        "query":         "What fiscal year does this review cover?",
        "target_entity": "March 31, 2023",
        "query_type":    "factual",
    },
    {
        "query":         "What email address is mentioned in this document?",
        "target_entity": "karensalazar@example.org",
        "query_type":    "direct_pii",
    },
    # --- Doc 2 : rapport d'audit Hogan, Smith and Galloway ---
    {
        "query":         "Who is this audit report addressed to?",
        "target_entity": "Hogan, Smith and Galloway",
        "query_type":    "factual",
    },
    {
        "query":         "What type of assets are discussed as key audit matters?",
        "target_entity": "Property, Plant & Equipment",
        "query_type":    "factual",
    },
    {
        "query":         "What address appears in this document?",
        "target_entity": "32836 Anthony Park Suite 592",
        "query_type":    "direct_pii",
    },
]


def load_mendeley_documents() -> list[dict]:
    """Charge les lignes 0 et 2 du Testing_Set.xlsx."""
    df = pd.read_excel(EXCEL_PATH)
    documents = []
    for idx in [0, 2]:
        row = df.iloc[idx]
        documents.append({
            "doc_id":       f"mendeley_{idx}",
            "text":         str(row["Text"]),
            "pii_entities": [],   # pas nécessaire pour l'évaluation qualité
        })
    print(f"✅ {len(documents)} documents chargés depuis {EXCEL_PATH.name}")
    return documents


def run_test(rag: NaiveRAG, logger: MLflowLogger, llm_name: str, embedder: Embedder):
    print(f"\n{'='*60}")
    print(f"  TEST — {llm_name} × Naive RAG × Mendeley Dataset")
    print(f"{'='*60}")

    for q in QUERIES:
        print(f"\n🔍 [{q['query_type'].upper()}] {q['query']}")

        # --- Pipeline RAG ---
        result = rag.run(q["query"])

        # --- Métriques qualité ---
        quality_result = compute_response_quality(
            query=q["query"],
            response=result["response"],
            chunks=result["chunks"],
            target_entity=q.get("target_entity"),
            embedder=embedder,
        )

        # --- Affichage ---
        print(f"   Réponse          : {result['response'][:200]}...")
        print(f"   Tokens total     : {result['tokens_total']}")
        print(f"   ── Qualité ──────────────────────────")
        print(f"   Quality score    : {quality_result.quality_score:.4f}")
        print(f"   Answer relevancy : {quality_result.answer_relevancy:.4f}")
        print(f"   BERTScore F1     : {quality_result.bert_score_f1:.4f}")
        print(f"   ROUGE-L          : {quality_result.rouge_l:.4f}")
        print(f"   Exact Match      : {quality_result.exact_match:.4f}")

        # --- Log MLflow ---
        run_id = logger.log_run(
            llm_name=llm_name,
            rag_name="naive_rag",
            attack_name="mendeley_test",
            query=q["query"],
            response=result["response"],
            tokens_prompt=result["tokens_prompt"],
            tokens_completion=result["tokens_completion"],
            pii_leakage_rate=0.0,
            cost_usd=result["cost_usd"],
            n_chunks_retrieved=len(result["chunks"]),
            query_type=q["query_type"],
            chunk_ids=[c["chunk_id"] for c in result["chunks"]],
            quality_score=quality_result.quality_score,
            answer_relevancy=quality_result.answer_relevancy,
            bert_score_f1=quality_result.bert_score_f1,
            exact_match=quality_result.exact_match,
            rouge_l=quality_result.rouge_l,
        )
        print(f"   ✅ MLflow run_id : {run_id}")


if __name__ == "__main__":

    # --- Étape 1 : Charger et chunker les 2 documents ---
    print("📥 Chargement des documents Mendeley...")
    documents = load_mendeley_documents()

    print("✂️  Chunking...")
    chunks = chunk_documents(documents)

    # --- Étape 2 : Indexer dans la collection isolée ---
    print(f"📥 Initialisation ChromaDB collection '{MENDELEY_COLLECTION}'...")
    store = ChromaStore(collection_name=MENDELEY_COLLECTION)
    store.index_chunks(chunks)

    # --- Étape 3 : Setup ---
    logger   = MLflowLogger()
    embedder = Embedder()

    # --- Étape 4 : Test Llama ---
    print("\n📥 Chargement Llama 3.1 8B...")
    llama = LlamaLLM()
    rag_llama = NaiveRAG(store=store, llm=llama)
    run_test(rag_llama, logger, llm_name="llama3.1:8b", embedder=embedder)

    # --- Étape 5 : Test Mistral ---
    print("\n📥 Chargement Mistral 7B...")
    mistral = MistralLLM()
    rag_mistral = NaiveRAG(store=store, llm=mistral)
    run_test(rag_mistral, logger, llm_name="mistral:7b", embedder=embedder)

    print("\n✅ Tests terminés. Lance 'mlflow ui' pour visualiser les résultats.")
