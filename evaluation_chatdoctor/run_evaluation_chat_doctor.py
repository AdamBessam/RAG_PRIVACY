"""
run_evaluation_chat_doctor.py — Orchestrateur d'évaluation Zhang et al. (IPM 2026)
appliqué au dataset médical ChatDoctor / HealthCareMagic.

Même pipeline que evaluation_zhang/run_evaluation.py, données et caches isolés :
  1. Charger le doc_index (300 dialogues) et les requêtes d'attaque
  2. CPB v3 (llama3.1:8b) → réponses
  3. Métriques privacy  : LO (ROUGE-L), AE (juge GPT-4o), PI (ChromaDB + GPT-4o)
  4. Métriques utilité  : CR, SS, AR (RAGAS + GPT-4o, réponses de référence GPT-4o)
  5. MLflow logging
  6. Tableau de résultats

Note d'implémentation : metric_pi.py (evaluation_zhang/) code en dur ses chemins de
cache vers data/zhang_eval/. Pour ne jamais écraser les résultats Zhang existants,
ce script réutilise uniquement les fonctions de calcul pur de evaluation_zhang/
(PIMetric._decompose_*, PIMetric._precompute_weights, PIMetric.compute_pi,
compute_lo, score_ae) et gère lui-même tous les fichiers de cache sous
data/chatdoctor_eval/. RAGAS (CR/SS/AR) est appelé directement (pas via
metric_utility.compute_utility) pour pouvoir le découper en lots cachés —
voir compute_utility_chunked.

Checkpointing : chaque boucle payante (AE, décomposition PI, scoring PI,
références, RAGAS) sauvegarde son résultat après CHAQUE instance/lot via
run_checkpointed — un crash en cours de route ne fait perdre que l'appel en
vol, pas toute l'étape (voir index_chunks.py pour l'indexation ChromaDB,
qui suit la même logique de reprise).

Usage:
  python run_evaluation_chat_doctor.py [--skip-generation]
    --skip-generation  : reuse responses already saved in data/chatdoctor_eval/responses.json
"""
# Assurance peu coûteuse contre les conflits de runtimes natifs (PyTorch MKL/OpenMP
# vs cœur Rust de ChromaDB) : on sérialise les pools de threads natifs. Le vrai
# correctif du crash est ailleurs — ce sont les IMPORTS PARESSEUX (voir plus bas) ;
# ces variables ne sont qu'un filet de sécurité, sans coût réel sur machine CPU.
from __future__ import annotations  # annotations en chaînes → PIMetric/chromadb restent en import paresseux

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")        # cœur Rust de ChromaDB
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# IMPORTANT — IMPORTS PARESSEUX (même principe que evaluation_zhang/run_evaluation.py,
# qui tourne sans crash). Les bibliothèques natives lourdes ne sont PAS importées
# au niveau module : chromadb, torch (sentence-transformers, tiré par dataset_prep
# et metric_pi), openai, numpy. Les charger toutes à l'import co-chargeait torch ET
# le cœur Rust de ChromaDB dans le même process AVANT tout ordre maîtrisé → c'est
# précisément ce qui déclenchait l'access violation natif sous Windows. On les
# importe donc À L'INTÉRIEUR des fonctions, au dernier moment, comme l'original.
# mlflow est lui aussi importé tardivement (cf. main()) : l'initialiser dans le
# même process que ChromaDB provoque le même crash natif.

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "evaluation_zhang"))

from config import LLAMA_MODEL, MLFLOW_TRACKING_URI, OPENAI_API_KEY

DATA_DIR = ROOT / "data" / "chatdoctor_eval"
CHROMA_CORPUS_DIR = ROOT / "data" / "chroma_chatdoctor"
CHROMA_CLAIMS_DIR = ROOT / "data" / "chroma_chatdoctor_claims"
RESPONSES_PATH = DATA_DIR / "responses.json"
CONTEXTS_PATH = DATA_DIR / "contexts.json"
REFERENCES_PATH = DATA_DIR / "reference_responses.json"
RESULTS_PATH = DATA_DIR / "results.json"
CSV_PATH = DATA_DIR / "results_per_query.csv"
EXPERIMENT_NAME = "chatdoctor_evaluation"

CORPUS_COLLECTION = "chatdoctor_eval_corpus"
CLAIMS_COLLECTION = "chatdoctor_claims"
CLAIMS_CACHE_PATH = DATA_DIR / "claims_cache.json"
CLAIMS_EMBEDDINGS_PATH = DATA_DIR / "claims_embeddings.npy"
INDEX_CHUNKS_SCRIPT = Path(__file__).parent / "index_chunks.py"
MAX_INDEXING_ATTEMPTS = 30

UTILITY_CHUNKS_DIR = DATA_DIR / "utility_chunks"
UTILITY_CHUNK_SIZE = 25

REFERENCE_PROMPT = """\
You are a medical expert. Based on the patient-doctor dialogue excerpt below, answer the \
following question accurately and completely.

Dialogue:
{document}

Question:
{query}

Answer:"""


# ── Checkpointed loop helper ─────────────────────────────────────────────────────

def run_checkpointed(items: list, cache_path: Path, compute_fn, desc: str) -> list:
    """
    Computes compute_fn(item) for each item, writing the growing result list to
    cache_path after EVERY item. A crash mid-loop (Ctrl+C, ChromaDB native
    crash, API error, power loss...) only loses the in-flight call — re-running
    resumes from len(cached results) instead of redoing the whole (paid) loop.
    """
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    start = len(results)
    total = len(items)
    if start >= total:
        print(f"  {desc}: already complete ({start}/{total})")
        return results
    if start > 0:
        print(f"  {desc}: resuming from {start}/{total}")

    for i in tqdm(range(start, total), desc=desc, initial=start, total=total):
        results.append(compute_fn(items[i]))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    return results


# ── ChromaStore wrapper ────────────────────────────────────────────────────────

class ChatDoctorChromaStore:
    """
    Wraps the chatdoctor_eval_corpus ChromaDB collection behind the ChromaStore
    interface expected by NaiveRAG and CPBNaiveRAGV3.
    """

    def __init__(self):
        # Imports paresseux (comme ZhangChromaStore) : chromadb + torch ne sont
        # chargés qu'ici, au dernier moment, jamais à l'import du module.
        import chromadb
        from chromadb.config import Settings
        from embeddings.embedder import Embedder

        self._embedder = Embedder()
        client = chromadb.PersistentClient(
            path=str(CHROMA_CORPUS_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(CORPUS_COLLECTION)
        print(f"ChatDoctorChromaStore ready: {self.collection.count()} chunks")

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


# ── CPB v3 inference ───────────────────────────────────────────────────────────

CPB_RESULTS_PATH = DATA_DIR / "cpb_results.json"  # checkpointed {response, contexts} per instance
BOOTSTRAP_REPORT_PATH = Path(__file__).parent / "bootstrap_report.json"  # domaine + catégories CPB


def log_and_save_bootstrap(result, path: Path = BOOTSTRAP_REPORT_PATH) -> None:
    """Affiche et sauvegarde ce que la contre-mesure CPB v3 a auto-découvert au
    démarrage : le domaine inféré, les catégories sensibles, et pour chaque
    catégorie avec quoi elle a été remplie (phrases de la taxonomie générées/
    réelles + types d'entités Presidio suggérés)."""
    categories = list(result.dynamic_categories)
    taxonomy = result.dynamic_taxonomy or {}
    hints = result.category_hints or {}

    report = {
        "domain": result.domain,
        "domain_confidence": round(float(result.domain_confidence), 4),
        "used_fallback": result.used_fallback,
        "learned_pii_types": sorted(result.learned_types),
        "categories": categories,
        "category_details": {
            cat: {
                "presidio_hints": sorted(hints.get(cat, [])),
                "n_phrases": len(taxonomy.get(cat, [])),
                "phrases": list(taxonomy.get(cat, [])),
            }
            for cat in categories
        },
    }

    print("\n── CPB v3 bootstrap (auto-découverte) ───────────────────────────")
    print(f"  Domaine décidé   : {result.domain}  (confiance {report['domain_confidence']:.2f}"
          + (", FALLBACK" if result.used_fallback else "") + ")")
    print(f"  Types PII appris : {len(report['learned_pii_types'])}")
    print(f"  Catégories ({len(categories)}) — avec quoi chacune a été remplie :")
    for cat in categories:
        d = report["category_details"][cat]
        hints_str = ", ".join(d["presidio_hints"]) or "—"
        examples = "; ".join(p[:60] for p in d["phrases"][:3]) or "—"
        print(f"    • {cat}")
        print(f"        hints Presidio : {hints_str}")
        print(f"        {d['n_phrases']} phrases — ex.: {examples}")
    print("─────────────────────────────────────────────────────────────────")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Rapport bootstrap sauvegardé → {path}\n")


def load_or_run_cpb(attacks: list[dict], skip_generation: bool) -> tuple[list[str], list[list[str]]]:
    if skip_generation and RESPONSES_PATH.exists() and CONTEXTS_PATH.exists():
        print("Loading cached responses...")
        with open(RESPONSES_PATH, encoding="utf-8") as f:
            responses = json.load(f)
        with open(CONTEXTS_PATH, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"{len(responses)} responses loaded from cache.")
        return responses, contexts_per_query

    # Génération CPB en process unique (comme evaluation_zhang/run_cpb_v3) — les
    # imports lourds restent paresseux. Checkpointée par instance via
    # run_checkpointed : un crash ne perd que l'appel en vol, relancer reprend.
    print("Running CPB v3 (llama3.1:8b)...")
    from countermeasure_v3.cpb_naive_rag_v3 import CPBNaiveRAGV3
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG

    store = ChatDoctorChromaStore()
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb = CPBNaiveRAGV3(naive_rag=naive_rag)
    log_and_save_bootstrap(cpb.bootstrap_result)

    def run_one(attack: dict) -> dict:
        result = cpb.run(attack["query"])
        return {
            "response": result["response"],
            "contexts": [c.get("text", "") for c in result.get("chunks", [])],
        }

    results = run_checkpointed(attacks, CPB_RESULTS_PATH, run_one, desc="CPB v3 generation")
    responses = [r["response"] for r in results]
    contexts_per_query = [r["contexts"] for r in results]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    with open(CONTEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)

    print(f"Responses saved → {RESPONSES_PATH}")
    return responses, contexts_per_query


# ── PI metric — own claims DB (isolated from evaluation_zhang/data/chroma_zhang_claims) ──

def build_claims_cache(pi_metric: PIMetric, doc_index: dict) -> None:
    """
    Decomposes the 300 ChatDoctor dialogues into atomic claims (GPT-4o, ~300
    paid calls — checkpointed per-doc via run_checkpointed) then embeds them
    (cheap/local, redone in full each time) and caches the final result:
      claims_cache.json       → {ids, texts, metadatas}
      claims_embeddings.npy   → float array (N, dim)

    This protects the GPT-4o spend from the ChromaDB insertion crash risk
    documented in index_chunks.py / README.md — if collection.add() crashes
    partway through, the (expensive) decomposition never has to be redone.
    """
    if CLAIMS_CACHE_PATH.exists() and CLAIMS_EMBEDDINGS_PATH.exists():
        with open(CLAIMS_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Claims cache already built: {len(cache['ids'])} claims")
        return

    decomposed_path = DATA_DIR / "claims_decomposed.json"
    doc_items = list(doc_index.items())

    def decompose_one(item: tuple[str, dict]) -> dict:
        doc_id, doc_data = item
        text = doc_data["text"]
        statements = pi_metric._decompose_document(text) if text.strip() else []
        return {"doc_id": doc_id, "statements": statements, "doc_length": len(text.split())}

    decomposed = run_checkpointed(doc_items, decomposed_path, decompose_one, desc="PI doc decompose (GPT-4o)")

    ids, statements, metadatas = [], [], []
    for entry in decomposed:
        for j, stmt in enumerate(entry["statements"]):
            ids.append(f"{entry['doc_id']}_claim_{j:04d}")
            statements.append(stmt)
            metadatas.append({"individual_id": entry["doc_id"], "doc_length": entry["doc_length"]})

    print(f"Embedding {len(statements)} claims...")
    import numpy as np
    from embeddings.embedder import Embedder
    embedder = Embedder()
    batch_size = 128
    embeddings = []
    for start in tqdm(range(0, len(statements), batch_size), desc="Embedding claims"):
        batch = statements[start : start + batch_size]
        embs = embedder.embed_texts(batch, batch_size=batch_size)
        embeddings.extend(embs.tolist())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLAIMS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "texts": statements, "metadatas": metadatas}, f, ensure_ascii=False)
    np.save(CLAIMS_EMBEDDINGS_PATH, np.array(embeddings, dtype=np.float32))
    print(f"Claims cache saved → {CLAIMS_CACHE_PATH} ({len(ids)} claims)")


def run_claims_indexing() -> None:
    """
    Inserts the cached claims into ChromaDB via the isolated index_chunks.py
    subprocess (same crash-avoidance + auto-retry strategy as dataset_prep.py's
    corpus indexing — see index_chunks.py docstring).
    """
    CHROMA_CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_INDEXING_ATTEMPTS + 1):
        print(f"Claims indexing attempt {attempt}/{MAX_INDEXING_ATTEMPTS}...")
        result = subprocess.run([
            sys.executable, str(INDEX_CHUNKS_SCRIPT),
            "--chroma-dir", str(CHROMA_CLAIMS_DIR),
            "--collection", CLAIMS_COLLECTION,
            "--chunks-cache", str(CLAIMS_CACHE_PATH),
            "--embeddings-cache", str(CLAIMS_EMBEDDINGS_PATH),
            "--insert-batch", "50",
        ])
        if result.returncode == 0:
            print("Claims indexing completed.")
            return
        print(f"  Claims indexing subprocess crashed (exit code {result.returncode}) — retrying...")

    raise RuntimeError(
        f"Failed to index claims after {MAX_INDEXING_ATTEMPTS} attempts. "
        "See index_chunks.py / README.md for the known ChromaDB Rust/Windows crash."
    )


def build_chatdoctor_claims_db(pi_metric: PIMetric, doc_index: dict) -> chromadb.Collection:
    """
    Decomposes the 300 ChatDoctor dialogues into atomic claims and indexes them
    into a dedicated ChromaDB collection, then precomputes uniqueness weights.
    Mirrors PIMetric.build_claims_db but writes to chroma_chatdoctor_claims
    instead of the hardcoded chroma_zhang_claims path.
    """
    import chromadb
    from chromadb.config import Settings

    build_claims_cache(pi_metric, doc_index)
    run_claims_indexing()

    client = chromadb.PersistentClient(
        path=str(CHROMA_CLAIMS_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(CLAIMS_COLLECTION)

    weights_flag = DATA_DIR / "claims_weights_computed.flag"
    if not weights_flag.exists():
        print("Precomputing uniqueness weights (offline)...")
        pi_metric._precompute_weights(collection)
        weights_flag.touch()
    else:
        print("Uniqueness weights already computed — skip")

    print(f"Claims DB ready: {collection.count()} claims")
    return collection


def compute_pi_scores(
    doc_index: dict,
    responses: list[str],
    attacks: list[dict],
    cache_path: Path,
) -> list[float]:
    # PI en process unique (comme evaluation_zhang). build_chatdoctor_claims_db
    # décompose les dialogues (GPT-4o, checkpointé), indexe les claims (via le
    # sous-processus index_chunks.py, même chemin éprouvé que le corpus) puis
    # calcule les poids d'unicité. Le scoring est checkpointé dans pi_scores.json :
    # supprimer ce fichier force un recalcul complet (la base de claims et les
    # poids restent en cache).
    from metric_pi import PIMetric

    pi_metric = PIMetric()
    claims_collection = build_chatdoctor_claims_db(pi_metric, doc_index)
    pi_metric._collection = claims_collection  # bypass des chemins codés en dur de metric_pi

    pairs = list(zip(responses, attacks))
    return run_checkpointed(
        pairs,
        cache_path,
        lambda pair: pi_metric.compute_pi(pair[0], pair[1]["doc_id"]),
        desc="PI response decompose+score (GPT-4o)",
    )


# ── Utility metric — own reference responses cache ─────────────────────────────

def generate_reference_responses(attacks: list[dict], doc_index: dict) -> list[str]:
    """GPT-4o generates one gold reference answer per attack query. Checkpointed per-instance."""
    import openai
    from metric_pi import GPT4O_MODEL as PI_GPT4O_MODEL

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def generate_one(attack: dict) -> str:
        doc_text = doc_index.get(attack["doc_id"], {}).get("text", "")[:3000]
        prompt = REFERENCE_PROMPT.format(document=doc_text, query=attack["query"])
        try:
            resp = client.chat.completions.create(
                model=PI_GPT4O_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[generation error: {e}]"

    return run_checkpointed(attacks, REFERENCES_PATH, generate_one, desc="Reference responses (GPT-4o)")


# ── Utility metrics — RAGAS in cached chunks (CR/SS/AR can be 1000+ GPT-4o calls) ──

def compute_utility_chunked(
    attacks: list[dict],
    responses: list[str],
    contexts_per_query: list[list[str]],
    references: list[str],
    chunk_size: int = UTILITY_CHUNK_SIZE,
) -> dict[str, float]:
    """
    Runs RAGAS (context_precision / answer_similarity / answer_relevancy) in
    small chunks, caching each chunk's per-row scores to disk. RAGAS's
    evaluate() is a single opaque call that can issue 1000+ GPT-4o requests
    for 300 instances (context_precision alone calls the LLM once per
    retrieved chunk) — chunking bounds how much is lost to a crash to one
    chunk instead of the entire metric.
    """
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from metric_pi import GPT4O_MODEL as PI_GPT4O_MODEL
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, answer_similarity, context_precision

    llm = LangchainLLMWrapper(ChatOpenAI(model=PI_GPT4O_MODEL, api_key=OPENAI_API_KEY, temperature=0))
    metrics = [context_precision, answer_similarity, answer_relevancy]
    for m in metrics:
        m.llm = llm

    UTILITY_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(attacks)
    n_chunks = (n + chunk_size - 1) // chunk_size
    all_rows: list[dict] = []

    for c in range(n_chunks):
        start, end = c * chunk_size, min((c + 1) * chunk_size, n)
        chunk_path = UTILITY_CHUNKS_DIR / f"chunk_{c:03d}.json"

        if chunk_path.exists():
            print(f"  RAGAS chunk {c + 1}/{n_chunks} ({start}-{end}): already cached")
            with open(chunk_path, encoding="utf-8") as f:
                all_rows.extend(json.load(f))
            continue

        print(f"  RAGAS chunk {c + 1}/{n_chunks} ({start}-{end})...")
        data = {
            "question": [a["query"] for a in attacks[start:end]],
            "answer": responses[start:end],
            "contexts": contexts_per_query[start:end],
            "ground_truth": references[start:end],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()

        chunk_rows = [
            {
                "CR": float(row["context_precision"]),
                "SS": float(row["answer_similarity"]),
                "AR": float(row["answer_relevancy"]),
            }
            for _, row in df.iterrows()
        ]
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk_rows, f, ensure_ascii=False)
        all_rows.extend(chunk_rows)

    return {
        "CR": sum(r["CR"] for r in all_rows) / len(all_rows),
        "SS": sum(r["SS"] for r in all_rows) / len(all_rows),
        "AR": sum(r["AR"] for r in all_rows) / len(all_rows),
    }


# ── Results table ──────────────────────────────────────────────────────────────

def print_results_table(metrics: dict) -> None:
    order = ["LO_F1", "AE", "PI", "CR", "SS", "AR"]
    directions = {"LO_F1": "↓", "AE": "↑", "PI": "↓", "CR": "↑", "SS": "↑", "AR": "↑"}

    print("\n" + "=" * 40)
    print("  RESULTS — CPB v3 on ChatDoctor (medical)")
    print("=" * 40)
    print(f"  {'Metric':<10} {'Dir':>4} {'CPB v3':>10}")
    print("-" * 40)
    for m in order:
        val = metrics.get(m)
        val_str = f"{val:.4f}" if val is not None else "   N/A"
        print(f"  {m:<10} {directions.get(m, ''):>4} {val_str:>10}")
    print("=" * 40)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(skip_generation: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=== ChatDoctor Evaluation Harness — CPB v3 (Zhang et al. protocol) ===\n")

    # 1. Load data
    print("1. Loading data...")
    from dataset_prep import prepare_dataset    # tire torch — import paresseux
    from attack_builder import prepare_attacks

    doc_index, _ = prepare_dataset()
    attacks = prepare_attacks(doc_index)
    print(f"   {len(doc_index)} documents, {len(attacks)} attack queries\n")

    # ── Phase 1: everything that touches ChromaDB directly (generation + PI). ──
    # mlflow must NOT be imported/initialized yet — see the NOTE at the top of
    # this file. All work below only reads/writes plain JSON caches + ChromaDB.

    # 2. Generate responses
    print("2. Generating CPB v3 responses...")
    responses, contexts_per_query = load_or_run_cpb(attacks, skip_generation)
    print()

    # 3. Privacy metrics
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
    import openai
    from metric_ae import aggregate_ae, score_ae
    ae_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    ae_results = run_checkpointed(
        list(zip(responses, attacks)),
        DATA_DIR / "ae_results.json",
        lambda pair: score_ae(ae_client, pair[0], pair[1]["privacy_info"]),
        desc="AE (GPT-4o)",
    )
    ae_score = aggregate_ae(ae_results)
    print(f"       AE={ae_score:.4f}")

    print("   [PI] Personal Identification...")
    from metric_pi import PIMetric
    pi_scores = compute_pi_scores(doc_index, responses, attacks, DATA_DIR / "pi_scores.json")
    pi_score = PIMetric.aggregate_pi(pi_scores)
    print(f"       PI={pi_score:.4f}")

    # 4. Utility metrics — RAGAS itself doesn't touch ChromaDB, but keep it in
    # Phase 1 too so mlflow stays uninitialized until truly everything is done.
    print("\n4. Utility metrics (RAGAS + GPT-4o, cached per chunk)...")
    references = generate_reference_responses(attacks, doc_index)
    utility = compute_utility_chunked(attacks, responses, contexts_per_query, references)
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

    cpb_metrics = {
        "LO_F1": lo_agg["f1"],
        "AE":    ae_score,
        "PI":    pi_score,
        "CR":    utility["CR"],
        "SS":    utility["SS"],
        "AR":    utility["AR"],
    }
    all_results = {
        "system":      "cpb_v3",
        "llm":         LLAMA_MODEL,
        "dataset":     "LinhDuong/chatdoctor-200k",
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

    # ── Phase 2: mlflow logging — ChromaDB is never touched again past this point. ──
    print("\n6. MLflow logging...")
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"cpb_v3_chatdoctor_{LLAMA_MODEL.replace(':', '_')}"):
        mlflow.log_param("system", "cpb_v3")
        mlflow.log_param("llm_generation", LLAMA_MODEL)
        mlflow.log_param("llm_evaluation", "gpt-4o")
        mlflow.log_param("dataset", "LinhDuong/chatdoctor-200k")
        mlflow.log_param("n_queries", len(attacks))
        mlflow.log_param("n_responses", len(responses))

        mlflow.log_metric("LO_precision", lo_agg["precision"])
        mlflow.log_metric("LO_recall", lo_agg["recall"])
        mlflow.log_metric("LO_f1", lo_agg["f1"])
        mlflow.log_metric("AE", ae_score)
        mlflow.log_metric("PI", pi_score)
        mlflow.log_metric("CR", utility["CR"])
        mlflow.log_metric("SS", utility["SS"])
        mlflow.log_metric("AR", utility["AR"])

        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(CSV_PATH))

    # 7. Results table
    print_results_table(cpb_metrics)

    print(f"\nDone. Full results → {RESULTS_PATH}")
    print(f"      Per-query CSV  → {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatDoctor evaluation harness for CPB v3 (Zhang et al. protocol)")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse cached responses from a previous run",
    )
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
