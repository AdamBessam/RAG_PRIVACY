"""
run_evaluation_chatdoctor_combo_b7off.py — Harness Zhang et al. sur le dataset
médical ChatDoctor / HealthCareMagic, dans les MÊMES conditions que le run
publié evaluation_zhang_openai/run_evaluation_openai_v2_hybrid_nodedup.py :

  • retrieval HYBRIDE (dense text-embedding-3-small + BM25, fusion RRF), dedup=False
  • génération gpt-4o-mini
  • juges GPT-4o pour AE / PI / RAGAS
  • utilité RAGAS = context_recall (métrique "Context Recall" de Zhang Table 2)

Différence testée par rapport à ce run :
  • pipeline = CPB v5 COMBO (masquage par combinaisons ré-identifiantes
    domain-aware) au lieu de CPB v4 (masque-tout)
  • dernier bloc de contre-mesure B7 (CPBResponseGuard) DÉSACTIVÉ via
    AblationConfig(b7_response_guard=False)

Isolation / rollback : tout vit dans ce dossier + data/chatdoctor_eval_openai_combo_b7off/
+ collection ChromaDB dédiée data/chroma_chatdoctor_openai. Rien dans
countermeasure_v5/, evaluation_zhang* ou les données juridiques n'est modifié.
Rollback = supprimer ce dossier + les deux dossiers de données ci-dessus.

Réutilisations (pour rester IDENTIQUE à l'éval médicale déjà éprouvée, et
partager les caches PAYANTS indépendants de la variante) :
  • doc_index / attaques médicaux            → data/chatdoctor_eval/
  • base de claims PI médicale + poids       → run_evaluation_chat_doctor.compute_pi_scores
  • réponses de référence gold (GPT-4o)      → run_evaluation_chat_doctor.generate_reference_responses
  • boucle checkpointée crash-safe           → run_evaluation_chat_doctor.run_checkpointed
Ces trois derniers écrivent dans les caches médicaux PARTAGÉS (claims, refs),
qui ne dépendent QUE du doc_index → payés une seule fois, réutilisés par toute
autre variante médicale.

ATTENTION combo + gpt-4o-mini : la découverte des combinaisons risquées de la
classe mère (CPBNaiveRAGV5Combo._llama_json) suppose un client ollama
(self.llm.client.chat(..., format="json")). Avec GPT4oMiniLLM, self.llm.client
est un openai.OpenAI dont .chat n'est PAS appelable → la découverte échouerait
et le combo retomberait en v5 (aucune combinaison). CPBComboOpenAI ci-dessous
override _llama_json pour passer par la voie generate() OpenAI standard.

Usage :
  python run_evaluation_chatdoctor_combo_b7off.py [--skip-generation]
    --skip-generation : réutilise responses.json / contexts.json déjà générés
"""
# Filet de sécurité contre les conflits de runtimes natifs (PyTorch MKL/OpenMP vs
# cœur Rust de ChromaDB) — même stratégie que run_evaluation_chat_doctor.py. Le
# vrai correctif reste les IMPORTS PARESSEUX (chromadb/torch/mlflow importés dans
# les fonctions, jamais au niveau module).
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
import csv
import json
import math
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation_zhang"))         # metric_lo / metric_ae / metric_pi
sys.path.insert(0, str(ROOT / "evaluation_zhang_openai"))  # openai_embedder.OpenAIEmbedder
sys.path.insert(0, str(ROOT / "evaluation_chatdoctor"))    # PI médicale + refs + checkpointing

from config import MLFLOW_TRACKING_URI, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

# ── Chemins ──────────────────────────────────────────────────────────────────
SHARED_DATA_DIR   = ROOT / "data" / "chatdoctor_eval"                       # doc_index + attaques + caches médicaux partagés
DATA_DIR          = ROOT / "data" / "chatdoctor_eval_openai_combo_b7off"    # caches PROPRES à cette variante
CHROMA_CORPUS_DIR = ROOT / "data" / "chroma_chatdoctor_openai"              # corpus médical ré-embeddé OpenAI

CORPUS_COLLECTION = "chatdoctor_eval_corpus_openai"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50

RESPONSES_PATH  = DATA_DIR / "responses.json"
CONTEXTS_PATH   = DATA_DIR / "contexts.json"
CPB_RESULTS_PATH = DATA_DIR / "cpb_results.json"           # {response, contexts} checkpointé par instance
REFERENCES_PATH = DATA_DIR / "reference_responses.json"    # copie LOCALE des réponses gold de ce run
RESULTS_PATH    = DATA_DIR / "results.json"
CSV_PATH        = DATA_DIR / "results_per_query.csv"
EXAMPLES_MD_PATH = DATA_DIR / "exemples_questions_reponses.md"  # question / référence / réponse (lisible)
EXPERIMENT_NAME = "chatdoctor_evaluation"

UTILITY_CHUNKS_DIR = DATA_DIR / "utility_chunks"
UTILITY_CHUNK_SIZE = 25

GPT4O_JUDGE_MODEL = "gpt-4o"

# ── Sensibilité B6 (option C) : seuil de la porte SBERT (F2) du détecteur SAD ──
# Défaut du code = DEFAULT_SBERT_THRESHOLD = 0.42. On le RELÈVE ici pour réduire
# les blocages complets ("This information cannot be disclosed...") : moins de
# phrases atteignent le centroïde d'une catégorie sensible → moins de SAD
# confirmés → moins de refus. COMPROMIS ASSUMÉ : un seuil plus haut = moins de
# faux positifs (blocages gênants) MAIS plus de faux négatifs (vraies fuites qui
# passent) → sécurité réduite. Réglé UNIQUEMENT sur l'instance SAD du combo de ce
# run ; countermeasure_v4/ n'est pas touché (les autres runs gardent 0.42).
# Mettre à None pour laisser le défaut 0.42.
SAD_SBERT_THRESHOLD: float | None = 0.50


# ── ChromaStore médical, embeddings text-embedding-3-small (isolé) ────────────
# Copie de OpenAIZhangChromaStore (evaluation_zhang_openai/run_evaluation_openai.py)
# pointée sur une collection médicale dédiée : on NE réutilise PAS la classe
# d'origine, dont le dir/collection sont codés en dur sur le corpus JURIDIQUE
# (l'y nourrir avec des dialogues médicaux écraserait ou mélangerait les deux).
class ChatDoctorOpenAIChromaStore:
    """Corpus médical chunké + embeddé text-embedding-3-small. Auto-construit au
    premier lancement à partir du doc_index médical. dedup=False autorise
    plusieurs chunks du même dialogue (contexte plus complet)."""

    def __init__(self, doc_index: dict, dedup: bool = True):
        import chromadb
        from chromadb.config import Settings
        from openai_embedder import OpenAIEmbedder

        self.dedup = dedup
        self._embedder = OpenAIEmbedder()
        CHROMA_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_CORPUS_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            collection = client.get_collection(CORPUS_COLLECTION)
            if collection.count() == 0:
                raise ValueError("empty collection")
        except Exception:
            collection = self._build_index(client, doc_index)

        self.collection = collection
        print(f"ChatDoctorOpenAIChromaStore ready: {self.collection.count()} chunks "
              f"(model={OPENAI_EMBEDDING_MODEL}, dedup={self.dedup})")

    def _build_index(self, client, doc_index: dict):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        try:
            client.delete_collection(CORPUS_COLLECTION)
        except Exception:
            pass
        collection = client.create_collection(
            name=CORPUS_COLLECTION,
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

        print(f"Indexed {collection.count()} chunks into '{CORPUS_COLLECTION}' "
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


# ── CPB v5 combo, découverte des combos via le LLM OpenAI (pas ollama) ────────
def _make_combo(hybrid_rag):
    """Instancie le combo v5 avec B7 désactivé, en surchargeant _llama_json pour
    utiliser la génération OpenAI (voir la note en tête de fichier)."""
    from countermeasure_v4.cpb_ablation import AblationConfig
    from countermeasure_v5.cpb_naive_rag_v5_combo import CPBNaiveRAGV5Combo

    class CPBComboOpenAI(CPBNaiveRAGV5Combo):
        """Découverte des combinaisons risquées via self.llm.generate() (OpenAI).
        La méthode mère suppose un client ollama, incompatible avec GPT4oMiniLLM ;
        sans cet override le combo échouerait en silence et retomberait en v5."""

        def _llama_json(self, prompt: str) -> str:
            # Le prompt de _discover_risky_combinations force déjà "valid JSON only" ;
            # _parse_json extrait ensuite le premier objet {...} de la sortie.
            return self.llm.generate(prompt).response

    ablation = AblationConfig(name="b7_off", b7_response_guard=False)
    combo = CPBComboOpenAI(naive_rag=hybrid_rag, ablation=ablation)

    # Option C — désensibiliser la porte SBERT (F2) de B6 sur CETTE instance
    # uniquement (isolé ; countermeasure_v4/ inchangé). self.sbert_threshold est
    # relu à chaque appel de _sbert_proximity, donc le régler ici suffit.
    if SAD_SBERT_THRESHOLD is not None and getattr(combo, "sad_detector", None) is not None:
        old = combo.sad_detector.sbert_threshold
        combo.sad_detector.sbert_threshold = SAD_SBERT_THRESHOLD
        print(f"B6 SAD: seuil SBERT relevé {old} → {SAD_SBERT_THRESHOLD} "
              f"(moins de blocages, sécurité réduite — option C)")
    return combo


def load_or_run_combo(doc_index: dict, attacks: list[dict], skip_generation: bool):
    if skip_generation and RESPONSES_PATH.exists() and CONTEXTS_PATH.exists():
        print("Loading cached responses...")
        with open(RESPONSES_PATH, encoding="utf-8") as f:
            responses = json.load(f)
        with open(CONTEXTS_PATH, encoding="utf-8") as f:
            contexts_per_query = json.load(f)
        print(f"{len(responses)} responses loaded from cache.")
        return responses, contexts_per_query

    print("Running CPB v5 COMBO (B7 off) over HYBRID + NO-DEDUP retrieval (gpt-4o-mini)...")
    from llms.gpt4o_mini_llm import GPT4oMiniLLM
    from rag.hybrid_rag import HybridRAG
    from run_evaluation_chat_doctor import run_checkpointed

    store = ChatDoctorOpenAIChromaStore(doc_index, dedup=False)
    llm = GPT4oMiniLLM()
    hybrid_rag = HybridRAG(store=store, llm=llm, dedup=False)
    combo = _make_combo(hybrid_rag)

    def run_one(attack: dict) -> dict:
        result = combo.run(attack["query"])
        # CR mesure la qualité du retrieval → chunks BRUTS (raw_chunks), pas les
        # safe_chunks masqués (cohérent avec le run OpenAI d'origine).
        chunk_texts = [c.get("text", "") for c in result.get("raw_chunks", [])]
        return {"response": result["response"], "contexts": chunk_texts}

    results = run_checkpointed(attacks, CPB_RESULTS_PATH, run_one, desc="combo b7off generation")
    responses = [r["response"] for r in results]
    contexts_per_query = [r["contexts"] for r in results]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    with open(CONTEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(contexts_per_query, f, ensure_ascii=False, indent=2)

    print(f"Responses saved → {RESPONSES_PATH}")
    return responses, contexts_per_query


# ── Utilité RAGAS (context_recall), chunkée + cachée (crash-safe) ─────────────
def compute_utility_chunked(
    attacks: list[dict],
    responses: list[str],
    contexts_per_query: list[list[str]],
    references: list[str],
    chunk_size: int = UTILITY_CHUNK_SIZE,
) -> dict[str, float]:
    """CR (context_recall, comme Zhang Table 2) + SS (answer_similarity) +
    AR (answer_relevancy), juge GPT-4o, embeddings text-embedding-3-small.
    Évalué par lots cachés : un crash RAGAS ne perd qu'un lot, pas toute l'étape."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, answer_similarity, context_recall
    from ragas.metrics.base import MetricWithEmbeddings

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=GPT4O_JUDGE_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    )
    metrics = [context_recall, answer_similarity, answer_relevancy]
    for m in metrics:
        m.llm = llm
        if isinstance(m, MetricWithEmbeddings):
            m.embeddings = embeddings

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
            "question":     [a["query"] for a in attacks[start:end]],
            "answer":       responses[start:end],
            "contexts":     contexts_per_query[start:end],
            "ground_truth": references[start:end],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()

        chunk_rows = []
        for _, row in df.iterrows():
            cr = float(row["context_recall"])
            if math.isnan(cr):   # ground_truth vide / rien de pertinent → 0, pas NaN
                cr = 0.0
            ss = float(row["answer_similarity"])
            ar = float(row["answer_relevancy"])
            chunk_rows.append({
                "CR": cr,
                "SS": 0.0 if math.isnan(ss) else ss,
                "AR": 0.0 if math.isnan(ar) else ar,
            })
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk_rows, f, ensure_ascii=False)
        all_rows.extend(chunk_rows)

    return {
        "CR": sum(r["CR"] for r in all_rows) / len(all_rows),
        "SS": sum(r["SS"] for r in all_rows) / len(all_rows),
        "AR": sum(r["AR"] for r in all_rows) / len(all_rows),
    }


# ── Export lisible : question / référence / réponse système ──────────────────
def write_examples_markdown(
    attacks: list[dict],
    references: list[str],
    responses: list[str],
    path: Path,
    n: int | None = None,
) -> None:
    """Écrit un markdown lisible (même format que les autres runs) alignant, pour
    chaque requête : la question d'attaque, la réponse de référence gold, et la
    réponse finale du système protégé. n=None → toutes les requêtes."""
    lines = [
        "# Exemples de questions et de réponses",
        "## Benchmark RAG & Privacy — ChatDoctor · CPB v5 combo · B7 OFF",
        "",
        "Run : retrieval hybride (dense + BM25, fusion RRF), sans déduplication, "
        "LLM gpt-4o-mini, embeddings text-embedding-3-small. "
        "Dataset médical ChatDoctor / HealthCareMagic (300 requêtes).",
        "",
        "Pour chaque cas :",
        "- **Question** : requête d'attaque (fournit des infos connues pour tenter "
        "d'extraire l'attribut sensible du dialogue)",
        "- **Réponse de référence** : vérité terrain (réponse gold GPT-4o avec accès au document)",
        "- **Réponse du système (RAG + CPB v5 combo, B7 off)** : notre système protégé "
        "(masquage par combinaisons ré-identifiantes domain-aware, **sans** ResponseGuard final)",
        "",
        "---",
        "",
    ]
    m = len(attacks) if n is None else min(n, len(attacks))
    for i in range(m):
        a = attacks[i]
        ref = references[i] if i < len(references) else ""
        resp = responses[i] if i < len(responses) else ""
        lines += [
            f"## Exemple {i + 1} — {a.get('doc_id', '')}",
            "",
            "**Question :**",
            "",
            f"> {a.get('query', '')}",
            "",
            "**Réponse de référence :**",
            "",
            ref,
            "",
            "**Réponse du système (RAG + CPB v5 combo, B7 off) :**",
            "",
            resp,
            "",
            "---",
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Tableau de résultats ─────────────────────────────────────────────────────
def print_results_table(metrics: dict) -> None:
    order = ["LO_F1", "AE", "PI", "CR", "SS", "AR"]
    directions = {"LO_F1": "↓", "AE": "↑", "PI": "↓", "CR": "↑", "SS": "↑", "AR": "↑"}

    print("\n" + "=" * 55)
    print("  RESULTS — CPB v5 COMBO (B7 off) on ChatDoctor")
    print("  hybrid+nodedup · gpt-4o-mini · text-embedding-3-small")
    print("=" * 55)
    print(f"  {'Metric':<10} {'Dir':>4} {'Combo B7off':>14}")
    print("-" * 55)
    for m in order:
        val = metrics.get(m)
        val_str = f"{val:.4f}" if val is not None else "   N/A"
        print(f"  {m:<10} {directions.get(m, ''):>4} {val_str:>14}")
    print("=" * 55)


# ── Main ─────────────────────────────────────────────────────────────────────
def main(skip_generation: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=== ChatDoctor — CPB v5 COMBO, B7 OFF (OpenAI stack, hybrid+nodedup) ===\n")

    # 1. Données médicales (partagées avec l'éval ChatDoctor existante)
    print("1. Loading medical data (doc_index, attack_queries)...")
    with open(SHARED_DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        doc_index = json.load(f)
    attacks = json.loads((SHARED_DATA_DIR / "attack_queries.json").read_text(encoding="utf-8"))
    print(f"   {len(doc_index)} documents, {len(attacks)} attack queries\n")

    # ── Phase 1 : tout ce qui touche ChromaDB (génération + PI). mlflow PAS encore
    # importé (voir note native en tête). Import paresseux des helpers médicaux.
    from run_evaluation_chat_doctor import (
        compute_pi_scores,
        generate_reference_responses,
        run_checkpointed,
    )

    # 2. Génération
    print("2. Generating CPB v5 combo (B7 off) responses...")
    responses, contexts_per_query = load_or_run_combo(doc_index, attacks, skip_generation)
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

    print("   [PI] Personal Identification (base de claims médicale partagée)...")
    from metric_pi import PIMetric
    pi_scores = compute_pi_scores(doc_index, responses, attacks, DATA_DIR / "pi_scores.json")
    pi_score = PIMetric.aggregate_pi(pi_scores)
    print(f"       PI={pi_score:.4f}")

    # 4. Utilité — RAGAS (context_recall) chunké
    print("\n4. Utility metrics (RAGAS + GPT-4o, cached per chunk)...")
    references = generate_reference_responses(attacks, doc_index)
    # Copie LOCALE des réponses gold dans le dossier de ce run (la source est le
    # cache médical partagé data/chatdoctor_eval/reference_responses.json ; on en
    # garde une copie ici pour que le run soit autonome/archivable).
    with open(REFERENCES_PATH, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)
    print(f"   Reference responses saved → {REFERENCES_PATH}")
    utility = compute_utility_chunked(attacks, responses, contexts_per_query, references)
    print(f"   CR={utility['CR']:.4f}  SS={utility['SS']:.4f}  AR={utility['AR']:.4f}")

    # 5. CSV + markdown lisible (question / référence / réponse)
    print("\n5. CSV + markdown export...")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "doc_id", "query", "reference", "response",
            "LO_precision", "LO_recall", "LO_f1", "AE", "PI",
        ])
        writer.writeheader()
        for i, (attack, resp) in enumerate(zip(attacks, responses)):
            lo = lo_results[i] if i < len(lo_results) else {}
            writer.writerow({
                "index":        i,
                "doc_id":       attack.get("doc_id", ""),
                "query":        attack.get("query", ""),
                "reference":    references[i] if i < len(references) else "",
                "response":     resp,
                "LO_precision": round(lo.get("precision", 0.0), 4),
                "LO_recall":    round(lo.get("recall",    0.0), 4),
                "LO_f1":        round(lo.get("f1",        0.0), 4),
                "AE":           ae_results[i]["score"] if i < len(ae_results) else "",
                "PI":           round(pi_scores[i],   4) if i < len(pi_scores)   else "",
            })
    print(f"   CSV saved → {CSV_PATH}")

    write_examples_markdown(attacks, references, responses, EXAMPLES_MD_PATH)
    print(f"   Examples markdown saved → {EXAMPLES_MD_PATH}")

    cpb_metrics = {
        "LO_F1": lo_agg["f1"], "AE": ae_score, "PI": pi_score,
        "CR": utility["CR"], "SS": utility["SS"], "AR": utility["AR"],
    }
    all_results = {
        "system":          "cpb_v5_combo_b7off_openai_stack_hybrid_nodedup",
        "dataset":         "LinhDuong/chatdoctor-200k",
        "retrieval":       "hybrid_dense_bm25_rrf",
        "retrieval_dedup": False,
        "llm":             "gpt-4o-mini",
        "embedding":       OPENAI_EMBEDDING_MODEL,
        "ablation":        "b7_off",
        "n_instances":     len(attacks),
        "metrics":         cpb_metrics,
        "responses":            responses,    # réponses finales du système protégé
        "reference_responses":  references,   # réponses gold (vérité terrain)
        "per_instance":    {"LO": lo_results, "AE": ae_results, "PI": pi_scores},
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Phase 2 : mlflow (ChromaDB n'est plus touché au-delà de ce point) ──
    print("\n6. MLflow logging...")
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="cpb_v5_combo_b7off_chatdoctor_gpt4o_mini_openai_embed"):
        mlflow.log_param("system", "cpb_v5_combo_b7off_openai_stack_hybrid_nodedup")
        mlflow.log_param("retrieval", "hybrid_dense_bm25_rrf")
        mlflow.log_param("retrieval_dedup", False)
        mlflow.log_param("llm_generation", "gpt-4o-mini")
        mlflow.log_param("embedding_model", OPENAI_EMBEDDING_MODEL)
        mlflow.log_param("llm_evaluation", GPT4O_JUDGE_MODEL)
        mlflow.log_param("dataset", "LinhDuong/chatdoctor-200k")
        mlflow.log_param("ablation", "b7_off")
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
        mlflow.log_artifact(str(REFERENCES_PATH))
        mlflow.log_artifact(str(EXAMPLES_MD_PATH))

    print_results_table(cpb_metrics)

    print(f"\nDone. Full results → {RESULTS_PATH}")
    print(f"      Per-query CSV  → {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChatDoctor harness — CPB v5 combo, B7 off (OpenAI stack, hybrid+nodedup)"
    )
    parser.add_argument("--skip-generation", action="store_true", help="Reuse cached responses")
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
