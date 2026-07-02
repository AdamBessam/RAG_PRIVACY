"""
run_evaluation_openai.py — Variante "stack OpenAI" du harness Zhang et al.

Même pipeline que evaluation_zhang/run_evaluation.py (CPB v3, mêmes 300
requêtes d'attaque, mêmes métriques) mais avec :
  - Embedding   : text-embedding-3-small (OpenAI) au lieu de all-MiniLM-L6-v2 (local)
  - Génération  : gpt-4o-mini (OpenAI) au lieu de llama3.1:8b (Ollama local)

doc_index.json, attack_queries.json et reference_responses.json sont partagés
avec le run Llama (data/zhang_eval/) car ils ne dépendent pas du LLM/embedding
testé (mêmes docs, mêmes questions, mêmes réponses gold). Les fichiers propres
à cette variante (réponses CPB, contextes, scores AE/PI/utility) sont écrits
dans data/zhang_eval_openai/ pour ne pas écraser le run Llama et permettre la
comparaison des deux runs.

Usage:
  python run_evaluation_openai.py [--skip-generation]
    --skip-generation  : réutilise les réponses déjà sauvegardées dans
                          data/zhang_eval_openai/responses.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_zhang"))

from config import MLFLOW_TRACKING_URI, OPENAI_EMBEDDING_MODEL

# ── Published results (Zhang et al. Table 2) — fill in manually ───────────────
ZHANG_TABLE_2 = {
    "LO_F1": None,   # TODO
    "AE":    None,   # TODO
    "PI":    None,   # TODO
    "CR":    None,   # TODO
    "SS":    None,   # TODO
    "AR":    None,   # TODO
}

SHARED_DATA_DIR   = Path(__file__).parent.parent / "data" / "zhang_eval"          # doc_index, attack_queries, reference_responses (partagés avec le run Llama)
DATA_DIR          = Path(__file__).parent.parent / "data" / "zhang_eval_openai"   # responses, contexts, scores (propres à cette variante)
CHROMA_OPENAI_DIR = Path(__file__).parent.parent / "data" / "chroma_zhang_openai"

RESPONSES_PATH = DATA_DIR / "responses.json"
CONTEXTS_PATH  = DATA_DIR / "contexts.json"
RESULTS_PATH   = DATA_DIR / "results.json"
CSV_PATH       = DATA_DIR / "results_per_query.csv"
EXPERIMENT_NAME = "zhang_evaluation"

COLLECTION_NAME = "zhang_eval_corpus_openai"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50


# ── ChromaStore wrapper (text-embedding-3-small) ──────────────────────────────

class OpenAIZhangChromaStore:
    """
    Variante de ZhangChromaStore (evaluation_zhang/run_evaluation.py) avec
    text-embedding-3-small (OpenAI) au lieu de all-MiniLM-L6-v2 (local).
    Construit sa propre collection ChromaDB : espace vectoriel différent
    (1536-dim vs 384-dim), incompatible avec zhang_eval_corpus.
    """

    def __init__(self, doc_index: dict, dedup: bool = True):
        import chromadb
        from chromadb.config import Settings

        from openai_embedder import OpenAIEmbedder

        # dedup=True (défaut) : 1 chunk par document (cadre attaque/PI de Zhang).
        # dedup=False : top_k chunks bruts, plusieurs du même doc autorisés →
        # meilleure couverture du doc source (CR/SS/AR), plus de contexte au LLM.
        self.dedup = dedup
        self._embedder = OpenAIEmbedder()
        CHROMA_OPENAI_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_OPENAI_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            collection = client.get_collection(COLLECTION_NAME)
            if collection.count() == 0:
                raise ValueError("empty collection")
        except Exception:
            collection = self._build_index(client, doc_index)

        self.collection = collection
        print(f"OpenAIZhangChromaStore ready: {self.collection.count()} chunks "
              f"(model={OPENAI_EMBEDDING_MODEL}, dedup={self.dedup})")

    def _build_index(self, client, doc_index: dict):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        )

        ids, texts, metadatas = [], [], []
        for doc_id, doc_data in doc_index.items():
            text = doc_data.get("text", "")
            if not text.strip():
                continue
            for j, chunk in enumerate(splitter.split_text(text)):
                ids.append(f"{doc_id}_chunk_{j:04d}")
                texts.append(chunk)
                metadatas.append({
                    "source_doc_id": doc_id,
                    "chunk_index":   j,
                    "doc_id":        doc_id,
                    "n_pii":         0,
                    "pii_entities":  "[]",
                })

        print(f"Embedding {len(texts)} chunks with {OPENAI_EMBEDDING_MODEL}...")
        embeddings = self._embedder.embed_texts(texts).tolist()

        insert_batch = 100
        for start in range(0, len(ids), insert_batch):
            collection.add(
                ids=ids[start : start + insert_batch],
                embeddings=embeddings[start : start + insert_batch],
                documents=texts[start : start + insert_batch],
                metadatas=metadatas[start : start + insert_batch],
            )

        print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' "
              f"({self._embedder.total_tokens} embedding tokens, "
              f"${self._embedder.total_cost_usd:.4f})")
        return collection

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

            # dedup=False : on autorise plusieurs chunks du même document.
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


# ── CPB v3 inference (gpt-4o-mini) ────────────────────────────────────────────

def run_cpb_v3(doc_index: dict, attacks: list[dict]) -> tuple[list[str], list[list[str]]]:
    """
    Instantiates CPB v3 with gpt-4o-mini and text-embedding-3-small retrieval,
    runs it on all attack queries.
    Returns (responses, contexts_per_query).
    """
    from countermeasure_v3.cpb_naive_rag_v3 import CPBNaiveRAGV3
    from llms.gpt4o_mini_llm import GPT4oMiniLLM
    from rag.naive_rag import NaiveRAG

    store = OpenAIZhangChromaStore(doc_index)
    llm = GPT4oMiniLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV3(naive_rag=naive_rag)

    responses: list[str] = []
    contexts_per_query: list[list[str]] = []

    for i, attack in enumerate(attacks):
        print(f"  CPB v3 [{i + 1}/{len(attacks)}] {attack['doc_id']}...", end="\r")
        result = cpb.run(attack["query"])
        responses.append(result["response"])
        # CR mesure la qualité du retrieval → on évalue les chunks BRUTS récupérés,
        # pas les safe_chunks masqués (le masquage PII ne change pas quels docs sont
        # récupérés, seulement leur contenu ; l'évaluer masqué mélangerait qualité de
        # retrieval et dégâts du masquage).
        chunk_texts = [c.get("text", "") for c in result.get("raw_chunks", [])]
        contexts_per_query.append(chunk_texts)

    print()
    return responses, contexts_per_query


def load_or_run_cpb(doc_index: dict, attacks: list[dict], skip_generation: bool) -> tuple[list[str], list[list[str]]]:
    if skip_generation and RESPONSES_PATH.exists() and CONTEXTS_PATH.exists():
        print("Loading cached responses...")
        with open(RESPONSES_PATH, encoding="utf-8") as f:
            responses = json.load(f)
        with open(CONTEXTS_PATH, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"{len(responses)} responses loaded from cache.")
        return responses, contexts_per_query

    print("Running CPB v3 (gpt-4o-mini + text-embedding-3-small)...")
    responses, contexts_per_query = run_cpb_v3(doc_index, attacks)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    with open(CONTEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)

    print(f"Responses saved → {RESPONSES_PATH}")
    return responses, contexts_per_query


# ── Results table ──────────────────────────────────────────────────────────────

def print_results_table(cpb_metrics: dict) -> None:
    order = ["LO_F1", "AE", "PI", "CR", "SS", "AR"]
    directions = {"LO_F1": "↓", "AE": "↑", "PI": "↓", "CR": "↑", "SS": "↑", "AR": "↑"}

    print("\n" + "=" * 70)
    print("  RESULTS — CPB v3 (gpt-4o-mini + text-embedding-3-small)  vs  Zhang et al. Table 2")
    print("=" * 70)
    print(f"  {'Metric':<10} {'Dir':>4} {'CPB v3':>10} {'Zhang':>12} {'Delta':>10}")
    print("-" * 70)
    for m in order:
        our = cpb_metrics.get(m)
        zhang = ZHANG_TABLE_2.get(m)
        dir_str = directions.get(m, "")
        our_str = f"{our:.4f}" if our is not None else "   N/A"
        zhang_str = f"{zhang:.4f}" if zhang is not None else "  TODO"
        delta_str = f"{our - zhang:+.4f}" if (our is not None and zhang is not None) else "   N/A"
        print(f"  {m:<10} {dir_str:>4} {our_str:>10} {zhang_str:>12} {delta_str:>10}")
    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(skip_generation: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Zhang et al. Evaluation Harness — CPB v3 (OpenAI stack) ===\n")

    # 1. Load shared data (identique au run Llama, indépendant du LLM/embedding)
    print("1. Loading shared data (doc_index, attack_queries)...")
    with open(SHARED_DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        doc_index = json.load(f)
    attacks = json.loads((SHARED_DATA_DIR / "attack_queries.json").read_text(encoding="utf-8"))
    print(f"   {len(doc_index)} documents, {len(attacks)} attack queries\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="cpb_v3_gpt4o_mini_openai_embed"):
        mlflow.log_param("system", "cpb_v3_openai_stack")
        mlflow.log_param("llm_generation", "gpt-4o-mini")
        mlflow.log_param("embedding_model", OPENAI_EMBEDDING_MODEL)
        mlflow.log_param("llm_evaluation", "gpt-4o")
        mlflow.log_param("dataset", "umarbutler/open-australian-legal-corpus")
        mlflow.log_param("n_queries", len(attacks))

        # 2. Generate responses
        print("2. Generating CPB v3 responses...")
        responses, contexts_per_query = load_or_run_cpb(doc_index, attacks, skip_generation)
        mlflow.log_param("n_responses", len(responses))
        print()

        # 3. Privacy metrics (mêmes juges/formules que evaluation_zhang/run_evaluation.py)
        print("3. Privacy metrics")

        print("   [LO] ROUGE-L...")
        from metric_lo import aggregate_lo, compute_lo
        lo_results = [
            compute_lo(resp, doc_index.get(attack["doc_id"], {}).get("text", ""))
            for resp, attack in zip(responses, attacks)
        ]
        lo_agg = aggregate_lo(lo_results)
        print(f"       P={lo_agg['precision']:.4f}  R={lo_agg['recall']:.4f}  F1={lo_agg['f1']:.4f}")

        print("   [AE] GPT-4o judge...")
        from metric_ae import aggregate_ae, compute_ae_batch
        ae_cache_path = DATA_DIR / "ae_results.json"
        if ae_cache_path.exists():
            print("       Loading AE from cache...")
            with open(ae_cache_path, encoding="utf-8") as f:
                ae_results = json.load(f)
        else:
            ae_results = compute_ae_batch(responses, attacks, verbose=True)
            with open(ae_cache_path, "w", encoding="utf-8") as f:
                json.dump(ae_results, f, ensure_ascii=False, indent=2)
        ae_score = aggregate_ae(ae_results)
        print(f"       AE={ae_score:.4f}")

        print("   [PI] Personal Identification...")
        from metric_pi import PIMetric
        pi_cache_path = DATA_DIR / "pi_scores.json"
        pi_metric = PIMetric()
        if pi_cache_path.exists():
            print("       Loading PI from cache...")
            with open(pi_cache_path, encoding="utf-8") as f:
                pi_scores = json.load(f)
        else:
            # build_claims_db réutilise le cache existant (claims_built.flag) :
            # la claims DB ne dépend que de doc_index, pas du LLM/embedding testé.
            pi_metric.build_claims_db(doc_index)
            pi_scores = pi_metric.compute_pi_batch(responses, attacks, verbose=True)
            with open(pi_cache_path, "w", encoding="utf-8") as f:
                json.dump(pi_scores, f, ensure_ascii=False, indent=2)
        pi_score = PIMetric.aggregate_pi(pi_scores)
        print(f"       PI={pi_score:.4f}")

        # 4. Utility metrics
        print("\n4. Utility metrics (RAGAS + GPT-4o)...")
        from metric_utility import compute_utility, generate_reference_responses
        utility_cache_path = DATA_DIR / "utility_scores.json"
        # generate_reference_responses lit/écrit dans evaluation_zhang/DATA_DIR
        # (data/zhang_eval/reference_responses.json) -> réutilisé tel quel.
        references = generate_reference_responses(attacks, doc_index)
        if utility_cache_path.exists():
            print("   Loading utility from cache...")
            with open(utility_cache_path, encoding="utf-8") as f:
                utility = json.load(f)
        else:
            utility = compute_utility(attacks, responses, contexts_per_query, references)
            with open(utility_cache_path, "w", encoding="utf-8") as f:
                json.dump(utility, f, ensure_ascii=False, indent=2)
        print(f"   CR={utility['CR']:.4f}  SS={utility['SS']:.4f}  AR={utility['AR']:.4f}")

        # 5. CSV export — one row per query
        print("\n5. CSV export...")
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "index", "doc_id", "query", "response",
                "LO_precision", "LO_recall", "LO_f1",
                "AE", "PI",
            ])
            writer.writeheader()
            for i, (attack, resp) in enumerate(zip(attacks, responses)):
                lo = lo_results[i] if i < len(lo_results) else {}
                writer.writerow({
                    "index":        i,
                    "doc_id":       attack.get("doc_id", ""),
                    "query":        attack.get("query", ""),
                    "response":     resp,
                    "LO_precision": round(lo.get("precision", 0.0), 4),
                    "LO_recall":    round(lo.get("recall",    0.0), 4),
                    "LO_f1":        round(lo.get("f1",        0.0), 4),
                    "AE":           ae_results[i]["score"] if i < len(ae_results) else "",
                    "PI":           round(pi_scores[i],   4) if i < len(pi_scores)   else "",
                })
        print(f"   CSV saved → {CSV_PATH}")

        # 6. MLflow logging
        print("\n6. MLflow logging...")
        mlflow.log_metric("LO_precision", lo_agg["precision"])
        mlflow.log_metric("LO_recall", lo_agg["recall"])
        mlflow.log_metric("LO_f1", lo_agg["f1"])
        mlflow.log_metric("AE", ae_score)
        mlflow.log_metric("PI", pi_score)
        mlflow.log_metric("CR", utility["CR"])
        mlflow.log_metric("SS", utility["SS"])
        mlflow.log_metric("AR", utility["AR"])

        cpb_metrics = {
            "LO_F1": lo_agg["f1"],
            "AE":    ae_score,
            "PI":    pi_score,
            "CR":    utility["CR"],
            "SS":    utility["SS"],
            "AR":    utility["AR"],
        }
        all_results = {
            "system":      "cpb_v3_openai_stack",
            "llm":         "gpt-4o-mini",
            "embedding":   OPENAI_EMBEDDING_MODEL,
            "n_instances": len(attacks),
            "metrics":     cpb_metrics,
            "per_instance": {
                "LO": lo_results,
                "AE": ae_results,
                "PI": pi_scores,
            },
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(CSV_PATH))

        # 7. Results table
        print_results_table(cpb_metrics)

    print(f"\nDone. Full results → {RESULTS_PATH}")
    print(f"      Per-query CSV  → {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zhang et al. evaluation harness for CPB v3 — OpenAI stack "
                     "(gpt-4o-mini generation + text-embedding-3-small retrieval)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse cached responses from a previous run",
    )
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
