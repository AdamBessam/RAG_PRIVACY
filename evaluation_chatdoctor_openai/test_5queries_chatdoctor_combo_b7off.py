"""
test_5queries_chatdoctor_combo_b7off.py — Test rapide (5 requêtes) du pipeline
AVANT de lancer les 300 requêtes de run_evaluation_chatdoctor_combo_b7off.py.

But : vérifier de bout en bout, à faible coût, que TOUT fonctionne —
  • corpus médical ré-embeddé text-embedding-3-small
  • retrieval hybride dense+BM25 (dedup=False)
  • CPB v5 COMBO avec B7 désactivé (via _make_combo du harness → même code que
    le run complet, y compris l'override _llama_json pour la découverte des
    combinaisons en stack OpenAI)
  • les 6 métriques (LO, AE, PI, CR, SS, AR)

Réutilise les 5 PREMIÈRES vraies requêtes du run complet (data/chatdoctor_eval/,
même ordre que les 300) → aucune régénération de questions.

Tout est ISOLÉ dans des dossiers de test dédiés ; rien du run complet n'est
touché. Références gold générées à la volée pour les 5 docs (5 appels GPT-4o).

Usage :
  cd evaluation_chatdoctor_openai
  python test_5queries_chatdoctor_combo_b7off.py
  python test_5queries_chatdoctor_combo_b7off.py --skip-generation
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))                       # racine
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_zhang"))  # métriques
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_zhang_openai"))  # OpenAIEmbedder
sys.path.insert(0, str(Path(__file__).parent))                              # _make_combo du harness

from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

N_TEST_QUERIES = 5
GPT4O_JUDGE_MODEL = "gpt-4o"

ROOT = Path(__file__).parent.parent
FULL_DATA_DIR          = ROOT / "data" / "chatdoctor_eval"                            # run complet (lecture seule)
TEST_DATA_DIR          = ROOT / "data" / "chatdoctor_eval_test_combo_b7off"           # sorties isolées de ce test
CHROMA_TEST_DIR        = ROOT / "data" / "chroma_chatdoctor_openai_test"
CHROMA_CLAIMS_TEST_DIR = ROOT / "data" / "chroma_chatdoctor_claims_openai_test"

COLLECTION_NAME = "chatdoctor_openai_test_corpus"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50

# Prompt de référence médical (identique à celui du run complet ChatDoctor).
REFERENCE_PROMPT = """\
You are a medical expert. Based on the patient-doctor dialogue excerpt below, answer the \
following question accurately and completely.

Dialogue:
{document}

Question:
{query}

Answer:"""


# ── Step 1 : 5 vraies requêtes du run complet (mêmes 5, même ordre) ───────────
def load_5_queries() -> tuple[dict, list[dict]]:
    with open(FULL_DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        full_doc_index = json.load(f)
    with open(FULL_DATA_DIR / "attack_queries.json", encoding="utf-8") as f:
        full_attacks = json.load(f)

    attacks = full_attacks[:N_TEST_QUERIES]
    doc_index = {a["doc_id"]: full_doc_index[a["doc_id"]] for a in attacks if a["doc_id"] in full_doc_index}
    return doc_index, attacks


# ── Step 2 : collection ChromaDB de test (text-embedding-3-small) ─────────────
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
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, texts, metadatas = [], [], []
    for doc_id, doc_data in doc_index.items():
        text = doc_data["text"]
        if not text.strip():
            continue
        for j, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{doc_id}_chunk_{j:04d}")
            texts.append(chunk)
            metadatas.append({
                "source_doc_id": doc_id, "chunk_index": j, "doc_id": doc_id,
                "n_pii": 0, "pii_entities": "[]",
            })

    embeddings = embedder.embed_texts(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Test ChromaDB ready: {collection.count()} chunks from {len(doc_index)} docs "
          f"({embedder.total_tokens} tokens, ${embedder.total_cost_usd:.5f})")


class TestChatDoctorStore:
    """Collection de test (text-embedding-3-small). dedup=False → plusieurs chunks/doc."""

    def __init__(self, dedup: bool = True):
        import chromadb
        from chromadb.config import Settings
        from openai_embedder import OpenAIEmbedder

        self.dedup = dedup
        self._embedder = OpenAIEmbedder()
        client = chromadb.PersistentClient(
            path=str(CHROMA_TEST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"TestChatDoctorStore ready: {self.collection.count()} chunks (dedup={self.dedup})")

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
            if self.dedup:
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


# ── Step 3 : CPB v5 combo (B7 off) sur retrieval hybride + nodedup ────────────
def run_combo_test(attacks: list[dict]) -> tuple[list[str], list[list[str]]]:
    from llms.gpt4o_mini_llm import GPT4oMiniLLM
    from rag.hybrid_rag import HybridRAG

    # Même construction que le run complet (combo + B7 off + override _llama_json).
    from run_evaluation_chatdoctor_combo_b7off import _make_combo

    store = TestChatDoctorStore(dedup=False)
    llm = GPT4oMiniLLM()
    hybrid_rag = HybridRAG(store=store, llm=llm, dedup=False)
    combo = _make_combo(hybrid_rag)

    responses, contexts_per_query = [], []
    for i, attack in enumerate(attacks):
        print(f"  combo B7off [{i + 1}/{len(attacks)}] query: {attack['query'][:60]}...")
        result = combo.run(attack["query"])
        responses.append(result["response"])
        contexts_per_query.append([c.get("text", "") for c in result.get("raw_chunks", [])])
        print(f"    -> response ({len(result['response'])} chars): {result['response'][:80]}...")

    return responses, contexts_per_query


# ── Step 4 : références gold à la volée (5 appels GPT-4o) ──────────────────────
def generate_test_references(attacks: list[dict], doc_index: dict) -> list[str]:
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    refs = []
    for a in attacks:
        doc = doc_index.get(a["doc_id"], {}).get("text", "")[:3000]
        prompt = REFERENCE_PROMPT.format(document=doc, query=a["query"])
        try:
            resp = client.chat.completions.create(
                model=GPT4O_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            refs.append(resp.choices[0].message.content.strip())
        except Exception as e:
            refs.append(f"[generation error: {e}]")
    return refs


# ── Step 5 : les 6 métriques (claims DB PI mini isolée) ───────────────────────
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
    from metric_utility import compute_utility  # context_recall (comme le run complet)

    results = {}

    print("\n[LO] ROUGE-L...")
    lo_results = [
        compute_lo(resp, doc_index.get(attack["doc_id"], {}).get("text", ""))
        for resp, attack in zip(responses, attacks)
    ]
    lo_agg = aggregate_lo(lo_results)
    results["LO_F1"] = lo_agg["f1"]
    print(f"  P={lo_agg['precision']:.4f}  R={lo_agg['recall']:.4f}  F1={lo_agg['f1']:.4f}")

    print("\n[AE] GPT-4o judge...")
    ae_results = compute_ae_batch(responses, attacks, verbose=True)
    ae_score = aggregate_ae(ae_results)
    results["AE"] = ae_score
    print(f"  AE={ae_score:.4f}")
    for ae, attack in zip(ae_results, attacks):
        print(f"  doc {attack['doc_id']}: score={ae['score']}  {ae['justification'][:80]}")

    print("\n[PI] Personal Identification (claims DB isolée, 5 docs médicaux)...")
    pi_metric = PIMetric()

    import chromadb as _chromadb
    from chromadb.config import Settings as _Settings

    CHROMA_CLAIMS_TEST_DIR.mkdir(parents=True, exist_ok=True)
    claims_client = _chromadb.PersistentClient(
        path=str(CHROMA_CLAIMS_TEST_DIR),
        settings=_Settings(anonymized_telemetry=False),
    )
    claims_collection_name = "chatdoctor_claims_openai_test"
    try:
        claims_client.delete_collection(claims_collection_name)
    except Exception:
        pass
    claims_collection = claims_client.create_collection(
        name=claims_collection_name, metadata={"hnsw:space": "cosine"},
    )

    from embeddings.embedder import Embedder  # claims DB = embedder local (méthodo Zhang)

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

    print("\n[Utility] RAGAS context_recall (références générées à la volée)...")
    try:
        utility = compute_utility(attacks, responses, contexts_per_query, references)
        results["CR"] = utility["CR"]
        results["SS"] = utility["SS"]
        results["AR"] = utility["AR"]
        print(f"  CR={utility['CR']:.4f}  SS={utility['SS']:.4f}  AR={utility['AR']:.4f}")
    except ImportError as e:
        print(f"  RAGAS not installed: {e}")
        results["CR"] = results["SS"] = results["AR"] = None

    return results


# ── Tableau ───────────────────────────────────────────────────────────────────
def print_results_table(metrics: dict) -> None:
    order = [("LO_F1", "↓"), ("AE", "↑"), ("PI", "↓"), ("CR", "↑"), ("SS", "↑"), ("AR", "↑")]
    print("\n" + "=" * 50)
    print("  TEST (5 requêtes) — CPB v5 COMBO, B7 off — ChatDoctor")
    print("=" * 50)
    print(f"  {'Metric':<10} {'Dir':>5} {'Combo B7off':>14}")
    print("-" * 50)
    for metric, direction in order:
        val = metrics.get(metric)
        val_str = f"{val:.4f}" if val is not None else "   N/A"
        print(f"  {metric:<10} {direction:>5} {val_str:>14}")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(skip_generation: bool = False) -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    responses_path = TEST_DATA_DIR / "responses_test.json"
    contexts_path = TEST_DATA_DIR / "contexts_test.json"
    references_path = TEST_DATA_DIR / "references_test.json"

    print("=" * 60)
    print("  TEST PIPELINE — CPB v5 COMBO (B7 off) — ChatDoctor — 5 requêtes")
    print("=" * 60)

    print("\n[1] Loading 5 real queries from the full run...")
    doc_index, attacks = load_5_queries()
    print(f"    doc_ids: {list(doc_index.keys())}")

    if skip_generation and responses_path.exists() and contexts_path.exists():
        print("\n[2-3] Loading cached responses...")
        with open(responses_path, encoding="utf-8") as f:
            responses = json.load(f)
        with open(contexts_path, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"    {len(responses)} responses loaded.")
    else:
        print("\n[2] Building test ChromaDB (text-embedding-3-small)...")
        build_test_chroma(doc_index)

        print(f"\n[3] Running CPB v5 combo B7off (gpt-4o-mini) on {len(attacks)} queries...")
        responses, contexts_per_query = run_combo_test(attacks)
        with open(responses_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        with open(contexts_path, "w", encoding="utf-8") as f:
            json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)
        print(f"\n    Responses saved -> {responses_path}")

    print("\n[4] Generating gold reference responses (GPT-4o, 5 docs)...")
    if skip_generation and references_path.exists():
        with open(references_path, encoding="utf-8") as f:
            references = json.load(f)
    else:
        references = generate_test_references(attacks, doc_index)
        with open(references_path, "w", encoding="utf-8") as f:
            json.dump(references, f, ensure_ascii=False, indent=2)

    print("\n[5] Computing all metrics...")
    metrics = compute_all_metrics(attacks, doc_index, responses, contexts_per_query, references)

    # Export lisible (question / référence / réponse), même format que le run complet.
    from run_evaluation_chatdoctor_combo_b7off import write_examples_markdown
    md_path = TEST_DATA_DIR / "exemples_questions_reponses.md"
    write_examples_markdown(attacks, references, responses, md_path)
    print(f"\nExamples markdown -> {md_path}")

    print_results_table(metrics)

    out = {"n_queries": len(attacks), "doc_ids": list(doc_index.keys()), "metrics": metrics}
    out_path = TEST_DATA_DIR / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Full test results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 5 requêtes — CPB v5 combo, B7 off (ChatDoctor, OpenAI stack)")
    parser.add_argument("--skip-generation", action="store_true", help="Reuse cached responses")
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
