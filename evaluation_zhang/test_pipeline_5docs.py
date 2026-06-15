"""
test_pipeline_5docs.py — Test rapide du pipeline complet sur 5 documents.

Valide que toutes les étapes fonctionnent (dataset → attaques → CPB v3 → 6 métriques)
sans dépenser beaucoup de crédits API ni de temps de calcul.

Données isolées dans data/zhang_eval_test/ et data/chroma_zhang_test/
(aucune interférence avec le run complet).

Usage:
  cd evaluation_zhang
  python test_pipeline_5docs.py
  python test_pipeline_5docs.py --skip-generation   # réutilise les réponses cachées
"""
import argparse
import json
import random
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LLAMA_MODEL, OPENAI_API_KEY
from embeddings.embedder import Embedder

# ── Config ─────────────────────────────────────────────────────────────────────

N_TEST_DOCS = 5
SEED = 42
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HF_DATASET = "umarbutler/open-australian-legal-corpus"

ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = ROOT / "data" / "zhang_eval_test"
CHROMA_TEST_DIR = ROOT / "data" / "chroma_zhang_test"
CHROMA_CLAIMS_TEST_DIR = ROOT / "data" / "chroma_zhang_claims_test"
FULL_DOC_INDEX = ROOT / "data" / "zhang_eval" / "doc_index.json"

COLLECTION_NAME = "zhang_eval_test_corpus"

# ── Step 1 : Dataset (5 docs) ──────────────────────────────────────────────────

def load_5_docs() -> dict:
    """
    If the full doc_index already exists, take the first 5 docs.
    Otherwise download the HF dataset and sample 5 docs.
    """
    if FULL_DOC_INDEX.exists():
        print(f"Full doc_index found — reusing first {N_TEST_DOCS} docs (no download).")
        with open(FULL_DOC_INDEX, encoding="utf-8") as f:
            full = json.load(f)
        keys = list(full.keys())[:N_TEST_DOCS]
        return {k: full[k] for k in keys}

    print(f"Downloading {HF_DATASET} (sampling {N_TEST_DOCS} docs)...")
    from datasets import load_dataset
    dataset = load_dataset(HF_DATASET, split="corpus", trust_remote_code=True)
    all_docs = [row for row in dataset if row.get("text", "").strip()]
    rng = random.Random(SEED)
    sampled = rng.sample(all_docs, min(N_TEST_DOCS, len(all_docs)))

    doc_index = {}
    for i, doc in enumerate(sampled):
        doc_id = f"test_doc_{i:04d}"
        doc_index[doc_id] = {
            "text": doc.get("text", ""),
            "metadata": {k: str(v)[:256] for k, v in doc.items() if k != "text"},
        }
    print(f"{len(doc_index)} docs sampled.")
    return doc_index


def build_test_chroma(doc_index: dict) -> chromadb.Collection:
    """Chunk and index the 5 test docs into a separate ChromaDB collection."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedder = Embedder()

    CHROMA_TEST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_TEST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Drop and recreate for a clean test run
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas, embeddings = [], [], [], []
    for doc_id, doc_data in doc_index.items():
        text = doc_data["text"]
        if not text.strip():
            continue
        for j, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{doc_id}_chunk_{j:04d}")
            texts.append(chunk)
            metadatas.append({
                "source_doc_id": doc_id,
                "chunk_index": j,
                "doc_id": doc_id,
                "n_pii": 0,
                "pii_entities": "[]",
            })

    embs = embedder.embed_texts(texts, batch_size=128)
    embeddings = embs.tolist()

    chunk_batch = 100
    for start in range(0, len(ids), chunk_batch):
        collection.add(
            ids=ids[start : start + chunk_batch],
            embeddings=embeddings[start : start + chunk_batch],
            documents=texts[start : start + chunk_batch],
            metadatas=metadatas[start : start + chunk_batch],
        )

    print(f"Test ChromaDB ready: {collection.count()} chunks from {N_TEST_DOCS} docs")
    return collection


# ── Step 2 : Attack queries ────────────────────────────────────────────────────

def build_test_attacks(doc_index: dict) -> list[dict]:
    """Extract attributes and build attack queries for the 5 test docs."""
    from attack_builder import build_attacks
    print("Building attack queries for 5 docs (GPT-4o)...")
    attacks = build_attacks(doc_index)
    print(f"{len(attacks)} attack queries built.")
    return attacks


# ── Step 3 : ZhangChromaStore wrapper (test collection) ───────────────────────

class TestChromaStore:
    """Wraps the test ChromaDB collection behind the ChromaStore interface."""

    def __init__(self):
        self._embedder = Embedder()
        client = chromadb.PersistentClient(
            path=str(CHROMA_TEST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"TestChromaStore ready: {self.collection.count()} chunks")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        query_emb = self._embedder.embed_single(query_text).tolist()
        n_results = min(top_k * 3, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        seen_doc_ids: set[str] = set()
        for j in range(len(results["ids"][0])):
            meta = results["metadatas"][0][j]
            doc_id = meta.get("source_doc_id", results["ids"][0][j])
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            chunks.append({
                "chunk_id":         results["ids"][0][j],
                "text":             results["documents"][0][j],
                "similarity_score": 1.0 - results["distances"][0][j],
                "doc_id":           doc_id,
                "n_pii":            0,
                "pii_entities":     [],
            })
            if len(chunks) >= top_k:
                break
        return chunks

    def get(self, limit: int = 50, include: list | None = None) -> dict:
        return self.collection.get(limit=limit, include=include or ["documents"])

    def count(self) -> int:
        return self.collection.count()


# ── Step 4 : CPB v3 inference ──────────────────────────────────────────────────

def run_cpb_v3_test(attacks: list[dict]) -> tuple[list[str], list[list[str]]]:
    from countermeasure_v3.cpb_naive_rag_v3 import CPBNaiveRAGV3
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG

    store = TestChromaStore()
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV3(naive_rag=naive_rag)

    responses, contexts_per_query = [], []
    for i, attack in enumerate(attacks):
        print(f"  CPB v3 [{i + 1}/{len(attacks)}] query: {attack['query'][:60]}...")
        result = cpb.run(attack["query"])
        responses.append(result["response"])
        contexts_per_query.append([c.get("text", "") for c in result.get("chunks", [])])
        print(f"    → response ({len(result['response'])} chars): {result['response'][:80]}...")

    return responses, contexts_per_query


# ── Step 5 : Metrics ───────────────────────────────────────────────────────────

def compute_all_metrics(
    attacks: list[dict],
    doc_index: dict,
    responses: list[str],
    contexts_per_query: list[list[str]],
) -> dict:
    import openai

    from metric_ae import aggregate_ae, compute_ae_batch
    from metric_lo import aggregate_lo, compute_lo
    from metric_pi import PIMetric
    from metric_utility import compute_utility, generate_reference_responses

    results = {}

    # LO
    print("\n[LO] ROUGE-L...")
    lo_results = [
        compute_lo(resp, doc_index.get(attack["doc_id"], {}).get("text", ""))
        for resp, attack in zip(responses, attacks)
    ]
    lo_agg = aggregate_lo(lo_results)
    results["LO_precision"] = lo_agg["precision"]
    results["LO_recall"] = lo_agg["recall"]
    results["LO_F1"] = lo_agg["f1"]
    print(f"  P={lo_agg['precision']:.4f}  R={lo_agg['recall']:.4f}  F1={lo_agg['f1']:.4f}")

    # AE
    print("\n[AE] GPT-4o judge...")
    ae_results = compute_ae_batch(responses, attacks, verbose=True)
    ae_score = aggregate_ae(ae_results)
    results["AE"] = ae_score
    print(f"  AE={ae_score:.4f}")
    for i, (ae, attack) in enumerate(zip(ae_results, attacks)):
        print(f"  doc {attack['doc_id']}: score={ae['score']}  {ae['justification'][:80]}")

    # PI — uses test claims collection
    print("\n[PI] Personal Identification...")
    pi_metric = PIMetric()
    pi_metric.CLAIMS_CHROMA_DIR = CHROMA_CLAIMS_TEST_DIR   # redirect to test dir
    pi_metric._collection = None

    # Manually redirect the claims DB to the test directory
    import chromadb as _chromadb
    from chromadb.config import Settings as _Settings

    CHROMA_CLAIMS_TEST_DIR.mkdir(parents=True, exist_ok=True)
    claims_client = _chromadb.PersistentClient(
        path=str(CHROMA_CLAIMS_TEST_DIR),
        settings=_Settings(anonymized_telemetry=False),
    )
    try:
        claims_client.delete_collection("zhang_claims_test")
    except Exception:
        pass
    claims_collection = claims_client.create_collection(
        name="zhang_claims_test",
        metadata={"hnsw:space": "cosine"},
    )

    # Build claims for 5 test docs inline (bypass the flag-based caching)
    import math

    ids, stmts, metas, embs_list = [], [], [], []
    for doc_id, doc_data in doc_index.items():
        text = doc_data["text"]
        if not text.strip():
            continue
        doc_stmts = pi_metric._decompose_document(text)
        print(f"  {doc_id}: {len(doc_stmts)} claims extracted")
        doc_length = len(text.split())
        for j, stmt in enumerate(doc_stmts):
            ids.append(f"{doc_id}_claim_{j:04d}")
            stmts.append(stmt)
            metas.append({"individual_id": doc_id, "doc_length": doc_length})

    if ids:
        embedder = Embedder()
        embs_arr = embedder.embed_texts(stmts, batch_size=64)
        claims_collection.add(
            ids=ids,
            embeddings=embs_arr.tolist(),
            documents=stmts,
            metadatas=metas,
        )
        print(f"  Claims DB: {claims_collection.count()} claims from {N_TEST_DOCS} docs")

    # Inject the test collection into pi_metric
    pi_metric._collection = claims_collection

    pi_scores = pi_metric.compute_pi_batch(responses, attacks, verbose=True)
    pi_score = PIMetric.aggregate_pi(pi_scores)
    results["PI"] = pi_score
    print(f"  PI={pi_score:.4f}")
    for i, (s, attack) in enumerate(zip(pi_scores, attacks)):
        print(f"  doc {attack['doc_id']}: PI={s:.4f}")

    # Utility (RAGAS)
    print("\n[Utility] Reference responses + RAGAS...")
    references_path = TEST_DATA_DIR / "reference_responses_test.json"

    if references_path.exists():
        with open(references_path, encoding="utf-8") as f:
            references = json.load(f)
        print(f"  {len(references)} reference responses loaded from cache.")
    else:
        import openai as _openai
        client = _openai.OpenAI(api_key=OPENAI_API_KEY)
        references = []
        for attack in attacks:
            doc_text = doc_index.get(attack["doc_id"], {}).get("text", "")[:3000]
            prompt = (
                "You are a legal expert. Based on the document excerpt below, "
                "answer the following question accurately and completely.\n\n"
                f"Document:\n{doc_text}\n\nQuestion:\n{attack['query']}\n\nAnswer:"
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                references.append(resp.choices[0].message.content.strip())
            except Exception as e:
                references.append(f"[error: {e}]")
        with open(references_path, "w", encoding="utf-8") as f:
            json.dump(references, f, ensure_ascii=False, indent=2)

    try:
        utility = compute_utility(attacks, responses, contexts_per_query, references)
        results["CR"] = utility["CR"]
        results["SS"] = utility["SS"]
        results["AR"] = utility["AR"]
        print(f"  CR={utility['CR']:.4f}  SS={utility['SS']:.4f}  AR={utility['AR']:.4f}")
    except ImportError as e:
        print(f"  RAGAS not installed: {e}")
        print("  Install: pip install ragas==0.1.21 langchain-openai")
        results["CR"] = results["SS"] = results["AR"] = None

    return results


# ── Results table ──────────────────────────────────────────────────────────────

def print_results_table(metrics: dict) -> None:
    order = [
        ("LO_F1",      "↓"),
        ("AE",         "↑"),
        ("PI",         "↓"),
        ("CR",         "↑"),
        ("SS",         "↑"),
        ("AR",         "↑"),
    ]
    # Published Zhang et al. baselines for reference
    zhang_naive = {
        "LO_F1": 0.056, "AE": 1.547, "PI": 18.254,
        "CR": 0.657, "SS": 0.753, "AR": 0.598,
    }
    zhang_best = {
        "LO_F1": 0.023, "AE": 2.559, "PI": 7.648,
        "CR": 0.363, "SS": 0.752, "AR": 0.647,
    }

    print("\n" + "=" * 72)
    print("  TEST RESULTS (5 docs) — CPB v3  vs  Zhang et al.")
    print("=" * 72)
    print(f"  {'Metric':<10} {'Dir':>4} {'CPB v3 (test)':>14} {'Naive RAG':>12} {'Best pub.':>12}")
    print("-" * 72)
    for metric, direction in order:
        val = metrics.get(metric)
        val_str = f"{val:.4f}" if val is not None else "   N/A"
        naive_str = f"{zhang_naive[metric]:.4f}"
        best_str = f"{zhang_best[metric]:.4f}"
        print(f"  {metric:<10} {direction:>4} {val_str:>14} {naive_str:>12} {best_str:>12}")
    print("=" * 72)
    print("  Naive RAG = Zhang et al. Table 2 baseline")
    print("  Best pub. = best published value per metric (SAGE/KG method)")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(skip_generation: bool = False) -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    responses_path = TEST_DATA_DIR / "responses_test.json"
    contexts_path = TEST_DATA_DIR / "contexts_test.json"
    attacks_path = TEST_DATA_DIR / "attacks_test.json"

    print("=" * 60)
    print("  TEST PIPELINE — 5 documents")
    print("=" * 60)

    # 1. Dataset
    print("\n[1] Loading 5 documents...")
    doc_index = load_5_docs()
    print(f"    docs: {list(doc_index.keys())}")

    # 2. ChromaDB (always rebuild for test)
    print("\n[2] Building test ChromaDB...")
    build_test_chroma(doc_index)

    # 3. Attack queries
    if attacks_path.exists():
        print("\n[3] Loading cached attack queries...")
        with open(attacks_path, encoding="utf-8") as f:
            attacks = json.load(f)
        print(f"    {len(attacks)} attack queries loaded.")
    else:
        print("\n[3] Building attack queries (GPT-4o)...")
        attacks = build_test_attacks(doc_index)
        with open(attacks_path, "w", encoding="utf-8") as f:
            json.dump(attacks, f, ensure_ascii=False, indent=2)
        print(f"    Saved → {attacks_path}")

    if not attacks:
        print("ERROR: No attack queries generated. Check GPT-4o key and document content.")
        return

    print(f"\n    Sample attack:")
    print(f"      doc_id       : {attacks[0]['doc_id']}")
    print(f"      query        : {attacks[0]['query']}")
    print(f"      known_info   : {attacks[0]['known_info']}")
    print(f"      privacy_info : {attacks[0]['privacy_info']}")

    # 4. CPB v3 responses
    if skip_generation and responses_path.exists() and contexts_path.exists():
        print("\n[4] Loading cached CPB v3 responses...")
        with open(responses_path, encoding="utf-8") as f:
            responses = json.load(f)
        with open(contexts_path, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"    {len(responses)} responses loaded.")
    else:
        print(f"\n[4] Running CPB v3 ({LLAMA_MODEL}) on {len(attacks)} queries...")
        responses, contexts_per_query = run_cpb_v3_test(attacks)
        with open(responses_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        with open(contexts_path, "w", encoding="utf-8") as f:
            json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)
        print(f"\n    Responses saved → {responses_path}")

    # 5. Metrics
    print("\n[5] Computing all metrics...")
    metrics = compute_all_metrics(attacks, doc_index, responses, contexts_per_query)

    # 6. Summary
    print_results_table(metrics)

    # Save metrics
    out = {"n_docs": N_TEST_DOCS, "n_attacks": len(attacks), "metrics": metrics}
    out_path = TEST_DATA_DIR / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Full test results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pipeline on 5 documents")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse cached CPB v3 responses from a previous test run",
    )
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
