"""
test_pipeline_5docs.py — Test rapide du pipeline complet sur 5 dialogues ChatDoctor.

Valide que toutes les étapes fonctionnent (dataset → attaques → CPB v3 → 6 métriques)
sans dépenser beaucoup de crédits API ni de temps de calcul.

Données isolées dans data/chatdoctor_eval_test/ et data/chroma_chatdoctor_test/
(aucune interférence avec le run complet sur 300 dialogues).

Usage:
  cd evaluation_chatdoctor
  python test_pipeline_5docs.py
  python test_pipeline_5docs.py --skip-generation   # réutilise les réponses cachées
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import argparse
import json
import random
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "evaluation_zhang"))

from config import LLAMA_MODEL, OPENAI_API_KEY
from embeddings.embedder import Embedder

from dataset_prep import build_dialogue, HF_DATASET, HF_SPLIT  # evaluation_chatdoctor/dataset_prep.py

# ── Config ─────────────────────────────────────────────────────────────────────

N_TEST_DOCS = 5
SEED = 42
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TEST_DATA_DIR = ROOT / "data" / "chatdoctor_eval_test"
CHROMA_TEST_DIR = ROOT / "data" / "chroma_chatdoctor_test"
CHROMA_CLAIMS_TEST_DIR = ROOT / "data" / "chroma_chatdoctor_claims_test"
FULL_DOC_INDEX = ROOT / "data" / "chatdoctor_eval" / "doc_index.json"

COLLECTION_NAME = "chatdoctor_eval_test_corpus"

# ── Step 1 : Dataset (5 docs) ──────────────────────────────────────────────────

def load_5_docs() -> dict:
    """
    If the full doc_index already exists, take the first 5 docs.
    Otherwise download the HF dataset and sample 5 dialogues.
    """
    if FULL_DOC_INDEX.exists():
        print(f"Full doc_index found — reusing first {N_TEST_DOCS} docs (no download).")
        with open(FULL_DOC_INDEX, encoding="utf-8") as f:
            full = json.load(f)
        keys = list(full.keys())[:N_TEST_DOCS]
        return {k: full[k] for k in keys}

    print(f"Downloading {HF_DATASET} (sampling {N_TEST_DOCS} dialogues)...")
    from datasets import load_dataset
    dataset = load_dataset(HF_DATASET, split=HF_SPLIT)
    all_docs = [row for row in dataset if build_dialogue(row).strip()]
    rng = random.Random(SEED)
    sampled = rng.sample(all_docs, min(N_TEST_DOCS, len(all_docs)))

    doc_index = {}
    for i, doc in enumerate(sampled):
        doc_id = f"test_doc_{i:04d}"
        doc_index[doc_id] = {
            "text": build_dialogue(doc),
            "metadata": {k: str(v)[:256] for k, v in doc.items() if k not in ("input", "output")},
        }
    print(f"{len(doc_index)} dialogues sampled.")
    return doc_index


def build_test_chroma(doc_index: dict) -> chromadb.Collection:
    """Chunk and index the 5 test dialogues into a separate ChromaDB collection."""
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

    ids, texts, metadatas = [], [], []
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

    print(f"Test ChromaDB ready: {collection.count()} chunks from {N_TEST_DOCS} dialogues")
    return collection


# ── Step 2 : Attack queries ────────────────────────────────────────────────────

def build_test_attacks(doc_index: dict) -> list[dict]:
    """Extract medical attributes and build attack queries for the 5 test dialogues."""
    from attack_builder import build_attacks
    print("Building attack queries for 5 dialogues (GPT-4o)...")
    attacks = build_attacks(doc_index)
    print(f"{len(attacks)} attack queries built.")
    return attacks


# ── Step 3 : ChromaStore wrapper (test collection) ─────────────────────────────

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
    from metric_ae import aggregate_ae, compute_ae_batch
    from metric_lo import aggregate_lo, compute_lo
    from metric_pi import PIMetric
    from metric_utility import compute_utility

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
    for ae, attack in zip(ae_results, attacks):
        print(f"  doc {attack['doc_id']}: score={ae['score']}  {ae['justification'][:80]}")

    # PI — uses test claims collection (bypasses metric_pi.py's hardcoded zhang dirs
    # and compute_pi_batch's debug/exit hook by calling compute_pi directly)
    print("\n[PI] Personal Identification...")
    pi_metric = PIMetric()

    CHROMA_CLAIMS_TEST_DIR.mkdir(parents=True, exist_ok=True)
    claims_client = chromadb.PersistentClient(
        path=str(CHROMA_CLAIMS_TEST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        claims_client.delete_collection("chatdoctor_claims_test")
    except Exception:
        pass
    claims_collection = claims_client.create_collection(
        name="chatdoctor_claims_test",
        metadata={"hnsw:space": "cosine"},
    )

    ids, stmts, metas = [], [], []
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
        print(f"  Claims DB: {claims_collection.count()} claims from {N_TEST_DOCS} dialogues")
        pi_metric._precompute_weights(claims_collection)

    pi_metric._collection = claims_collection

    pi_scores = [pi_metric.compute_pi(resp, attack["doc_id"]) for resp, attack in zip(responses, attacks)]
    pi_score = PIMetric.aggregate_pi(pi_scores)
    results["PI"] = pi_score
    print(f"  PI={pi_score:.4f}")
    for s, attack in zip(pi_scores, attacks):
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
                "You are a medical expert. Based on the patient-doctor dialogue excerpt below, "
                "answer the following question accurately and completely.\n\n"
                f"Dialogue:\n{doc_text}\n\nQuestion:\n{attack['query']}\n\nAnswer:"
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
        ("LO_F1", "↓"),
        ("AE",    "↑"),
        ("PI",    "↓"),
        ("CR",    "↑"),
        ("SS",    "↑"),
        ("AR",    "↑"),
    ]

    print("\n" + "=" * 50)
    print("  TEST RESULTS (5 dialogues) — CPB v3 on ChatDoctor")
    print("=" * 50)
    print(f"  {'Metric':<10} {'Dir':>4} {'CPB v3 (test)':>14}")
    print("-" * 50)
    for metric, direction in order:
        val = metrics.get(metric)
        val_str = f"{val:.4f}" if val is not None else "   N/A"
        print(f"  {metric:<10} {direction:>4} {val_str:>14}")
    print("=" * 50)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(skip_generation: bool = False) -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    responses_path = TEST_DATA_DIR / "responses_test.json"
    contexts_path = TEST_DATA_DIR / "contexts_test.json"
    attacks_path = TEST_DATA_DIR / "attacks_test.json"

    print("=" * 60)
    print("  TEST PIPELINE — 5 ChatDoctor dialogues")
    print("=" * 60)

    # 1. Dataset
    print("\n[1] Loading 5 dialogues...")
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
        print("ERROR: No attack queries generated. Check GPT-4o key and dialogue content.")
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
    parser = argparse.ArgumentParser(description="Test pipeline on 5 ChatDoctor dialogues")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse cached CPB v3 responses from a previous test run",
    )
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
