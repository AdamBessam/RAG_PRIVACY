"""
test_5queries_openai.py — Test rapide (5 requêtes) du pipeline OpenAI stack
(gpt-4o-mini + text-embedding-3-small) avant de lancer les 300 requêtes du
run complet.

Contrairement à evaluation_zhang/test_pipeline_5docs.py (qui ré-échantillonne
5 docs indépendamment), ce test réutilise les VRAIES données du run complet
(data/zhang_eval/doc_index.json, attack_queries.json, reference_responses.json)
— les 5 premiers documents qui ont une question associée. C'est donc un
aperçu fidèle des 300 requêtes, pas un échantillon à part.

Coûts limités à ce qui change réellement par rapport au run Llama existant :
  - embeddings (5 docs chunkés + 5 requêtes) en text-embedding-3-small
  - génération des 5 réponses en gpt-4o-mini
  - juges AE/PI/Utility sur ces 5 nouvelles réponses (les références gold et
    les questions sont réutilisées telles quelles, pas régénérées)

Données isolées dans data/zhang_eval_test_openai/ et data/chroma_zhang_openai_test/
(aucune interférence avec le run complet ni avec le test Llama existant).

Usage:
  cd evaluation_zhang_openai
  python test_5queries_openai.py
  python test_5queries_openai.py --skip-generation
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_zhang"))

N_TEST_QUERIES = 5

ROOT = Path(__file__).parent.parent
FULL_DATA_DIR           = ROOT / "data" / "zhang_eval"                      # run complet (partagé, lecture seule)
TEST_DATA_DIR           = ROOT / "data" / "zhang_eval_test_openai"          # sorties isolées de ce test
CHROMA_TEST_DIR         = ROOT / "data" / "chroma_zhang_openai_test"
CHROMA_CLAIMS_TEST_DIR  = ROOT / "data" / "chroma_zhang_claims_openai_test"

COLLECTION_NAME = "zhang_eval_openai_test_corpus"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50


# ── Step 1: pick 5 real queries from the full run ─────────────────────────────

def load_5_queries() -> tuple[dict, list[dict], list[str]]:
    """
    Takes the first N_TEST_QUERIES documents (doc_index order) that have a
    matching attack query in the full run. Returns (doc_index_subset, attacks,
    reference_responses) — genuine slices of the 300-query run.
    """
    with open(FULL_DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        full_doc_index = json.load(f)
    with open(FULL_DATA_DIR / "attack_queries.json", encoding="utf-8") as f:
        full_attacks = json.load(f)
    with open(FULL_DATA_DIR / "reference_responses.json", encoding="utf-8") as f:
        full_refs = json.load(f)

    attack_by_doc = {a["doc_id"]: (i, a) for i, a in enumerate(full_attacks)}

    chosen_doc_ids = []
    for doc_id in full_doc_index:
        if doc_id in attack_by_doc:
            chosen_doc_ids.append(doc_id)
        if len(chosen_doc_ids) == N_TEST_QUERIES:
            break

    doc_index = {d: full_doc_index[d] for d in chosen_doc_ids}
    attacks, references = [], []
    for d in chosen_doc_ids:
        idx, attack = attack_by_doc[d]
        attacks.append(attack)
        references.append(full_refs[idx])

    return doc_index, attacks, references


# ── Step 2: isolated ChromaDB collection (text-embedding-3-small) ────────────

def build_test_chroma(doc_index: dict) -> None:
    import chromadb
    from chromadb.config import Settings
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    from openai_embedder import OpenAIEmbedder

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedder = OpenAIEmbedder()

    CHROMA_TEST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_TEST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

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

    embeddings = embedder.embed_texts(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Test ChromaDB ready: {collection.count()} chunks from {len(doc_index)} docs "
          f"({embedder.total_tokens} tokens, ${embedder.total_cost_usd:.5f})")


class TestOpenAIChromaStore:
    """Wraps the test ChromaDB collection (text-embedding-3-small) behind the ChromaStore interface."""

    def __init__(self):
        import chromadb
        from chromadb.config import Settings

        from openai_embedder import OpenAIEmbedder

        self._embedder = OpenAIEmbedder()
        client = chromadb.PersistentClient(
            path=str(CHROMA_TEST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"TestOpenAIChromaStore ready: {self.collection.count()} chunks")

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


# ── Step 3: CPB v3 inference (gpt-4o-mini) ─────────────────────────────────────

def run_cpb_v3_test(attacks: list[dict]) -> tuple[list[str], list[list[str]]]:
    from countermeasure_v3.cpb_naive_rag_v3 import CPBNaiveRAGV3
    from llms.gpt4o_mini_llm import GPT4oMiniLLM
    from rag.naive_rag import NaiveRAG

    store = TestOpenAIChromaStore()
    llm = GPT4oMiniLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV3(naive_rag=naive_rag)

    responses, contexts_per_query = [], []
    for i, attack in enumerate(attacks):
        print(f"  CPB v3 [{i + 1}/{len(attacks)}] query: {attack['query'][:60]}...")
        result = cpb.run(attack["query"])
        responses.append(result["response"])
        contexts_per_query.append([c.get("text", "") for c in result.get("chunks", [])])
        print(f"    -> response ({len(result['response'])} chars): {result['response'][:80]}...")

    return responses, contexts_per_query


# ── Step 4: Metrics (LO, AE, PI, Utility) ──────────────────────────────────────

def compute_all_metrics(
    attacks: list[dict],
    doc_index: dict,
    responses: list[str],
    contexts_per_query: list[list[str]],
    references: list[str],
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

    # PI — claims DB isolée, reconstruite à chaque run (5 docs seulement, coût négligeable)
    print("\n[PI] Personal Identification (claims DB isolée, 5 docs)...")
    pi_metric = PIMetric()

    import chromadb as _chromadb
    from chromadb.config import Settings as _Settings

    CHROMA_CLAIMS_TEST_DIR.mkdir(parents=True, exist_ok=True)
    claims_client = _chromadb.PersistentClient(
        path=str(CHROMA_CLAIMS_TEST_DIR),
        settings=_Settings(anonymized_telemetry=False),
    )
    claims_collection_name = "zhang_claims_openai_test"
    try:
        claims_client.delete_collection(claims_collection_name)
    except Exception:
        pass
    claims_collection = claims_client.create_collection(
        name=claims_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    from embeddings.embedder import Embedder  # claims DB = toujours l'embedder local (méthodo Zhang et al., indépendant du stack testé)

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
        local_embedder = Embedder()
        embs_arr = local_embedder.embed_texts(stmts, batch_size=64)
        claims_collection.add(ids=ids, embeddings=embs_arr.tolist(), documents=stmts, metadatas=metas)
        print(f"  Claims DB: {claims_collection.count()} claims from {len(doc_index)} docs")

    pi_metric._collection = claims_collection
    pi_scores = pi_metric.compute_pi_batch(responses, attacks, verbose=True)
    pi_score = PIMetric.aggregate_pi(pi_scores)
    results["PI"] = pi_score
    print(f"  PI={pi_score:.4f}")
    for s, attack in zip(pi_scores, attacks):
        print(f"  doc {attack['doc_id']}: PI={s:.4f}")

    # Utility (RAGAS) — références gold réutilisées du run complet
    print("\n[Utility] RAGAS (références réutilisées du run complet)...")
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
    order = [("LO_F1", "down"), ("AE", "up"), ("PI", "down"), ("CR", "up"), ("SS", "up"), ("AR", "up")]
    # Baseline du test Llama 5-docs existant (data/zhang_eval_test/test_results.json)
    # NB: échantillon de 5 docs différent (sampling indépendant) -> comparaison indicative seulement.
    llama_test_baseline = {
        "LO_F1": 0.0178, "AE": 2.75, "PI": 2.55, "CR": 0.25, "SS": 0.82, "AR": 0.20,
    }

    print("\n" + "=" * 74)
    print("  TEST RESULTS (5 requêtes réelles) — CPB v3 OpenAI stack")
    print("=" * 74)
    print(f"  {'Metric':<10} {'Dir':>5} {'gpt4o-mini':>12} {'Llama (autre échant.)':>22}")
    print("-" * 74)
    for metric, direction in order:
        val = metrics.get(metric)
        val_str = f"{val:.4f}" if val is not None else "   N/A"
        base_str = f"{llama_test_baseline[metric]:.4f}"
        print(f"  {metric:<10} {direction:>5} {val_str:>12} {base_str:>22}")
    print("=" * 74)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(skip_generation: bool = False) -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    responses_path = TEST_DATA_DIR / "responses_test.json"
    contexts_path = TEST_DATA_DIR / "contexts_test.json"

    print("=" * 60)
    print("  TEST PIPELINE OpenAI stack — 5 requêtes")
    print("=" * 60)

    print("\n[1] Loading 5 real queries from the full run (doc_index, attack_queries, references)...")
    doc_index, attacks, references = load_5_queries()
    print(f"    doc_ids: {list(doc_index.keys())}")

    if skip_generation and responses_path.exists() and contexts_path.exists():
        print("\n[2-3] Loading cached CPB v3 responses (skip ChromaDB rebuild + generation)...")
        with open(responses_path, encoding="utf-8") as f:
            responses = json.load(f)
        with open(contexts_path, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"    {len(responses)} responses loaded.")
    else:
        print("\n[2] Building test ChromaDB (text-embedding-3-small)...")
        build_test_chroma(doc_index)

        print(f"\n[3] Running CPB v3 (gpt-4o-mini) on {len(attacks)} queries...")
        responses, contexts_per_query = run_cpb_v3_test(attacks)
        with open(responses_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        with open(contexts_path, "w", encoding="utf-8") as f:
            json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)
        print(f"\n    Responses saved -> {responses_path}")

    print("\n[4] Computing all metrics...")
    metrics = compute_all_metrics(attacks, doc_index, responses, contexts_per_query, references)

    print_results_table(metrics)

    out = {"n_queries": len(attacks), "doc_ids": list(doc_index.keys()), "metrics": metrics}
    out_path = TEST_DATA_DIR / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Full test results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pipeline OpenAI stack (gpt-4o-mini + text-embedding-3-small) on 5 real queries")
    parser.add_argument("--skip-generation", action="store_true", help="Reuse cached CPB v3 responses from a previous test run")
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
