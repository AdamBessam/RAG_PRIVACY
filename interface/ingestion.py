"""
Interface 1 — Ingestion + analyse B0 (LOGIQUE PURE, sans Streamlit).

Rôle : à partir d'une source de données (fichier CSV, dataset Hugging Face, ou
texte brut collé), découper le corpus en chunks, l'indexer dans une base
vectorielle ChromaDB ISOLÉE, puis lancer l'étape B0 de la countermeasure v6
(domaine + catégories + taxonomie + combinaisons ré-identifiantes).

Isolation (demande explicite) :
  • Ce module ne modifie AUCUN fichier existant. Il réutilise tels quels
    Embedder, NaiveRAG, LlamaLLM et CPBNaiveRAGV6 — aucune dépendance aux
    autres dossiers countermeasure* (v3/v4/v5/base).
  • L'indexation se fait dans une base Chroma DÉDIÉE (dossier + collection
    propres à l'interface), jamais la collection de benchmark `rag_benchmark`.
    Défaut : data/interface_ingestion/  ·  collection "interface_ingestion".

Imports lourds (chromadb, datasets, langchain, embedder, bootstrap) faits DANS
les fonctions — jamais au niveau module — pour rester cohérent avec le harness
et éviter le crash natif Windows au chargement paresseux.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Base Chroma isolée par défaut (ne touche pas data/chroma_db du benchmark).
DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "data" / "interface_ingestion"
DEFAULT_COLLECTION = "interface_ingestion"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des documents (3 sources)
# ══════════════════════════════════════════════════════════════════════════════

def _table_name(file_or_path) -> str:
    """Nom de fichier, que ce soit un chemin str/Path ou un UploadedFile Streamlit."""
    name = getattr(file_or_path, "name", None) or str(file_or_path)
    return name.lower()


def read_table(file_or_path, nrows: int | None = None):
    """Lit un CSV ou un XLSX en DataFrame selon l'extension du fichier.

    Accepte un chemin (str/Path) ou un objet fichier Streamlit (UploadedFile).
    XLSX nécessite openpyxl (`pip install openpyxl`).
    """
    import pandas as pd

    name = _table_name(file_or_path)
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file_or_path, nrows=nrows)
    return pd.read_csv(file_or_path, nrows=nrows)


def load_docs_from_csv(file_or_path, text_column: str, max_docs: int | None = None) -> list[dict]:
    """Charge des documents depuis un fichier tabulaire CSV ou XLSX (chemin ou
    objet fichier Streamlit).

    Chaque ligne non vide de `text_column` devient un document. Les autres
    colonnes sont conservées (tronquées) comme métadonnées.
    """
    df = read_table(file_or_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Colonne '{text_column}' absente du CSV. Colonnes disponibles : {list(df.columns)}"
        )

    docs: list[dict] = []
    for i, row in df.iterrows():
        text = str(row[text_column]) if row[text_column] is not None else ""
        if not text.strip():
            continue
        meta = {
            k: str(v)[:256]
            for k, v in row.items()
            if k != text_column and v is not None
        }
        docs.append({"doc_id": f"doc_{len(docs):04d}", "text": text, "metadata": meta})
        if max_docs and len(docs) >= max_docs:
            break
    if not docs:
        raise ValueError(f"Aucun texte non vide trouvé dans la colonne '{text_column}'.")
    return docs


def normalize_hf_name(value: str) -> str:
    """Accepte aussi bien un identifiant de dépôt (`ildpil/text-anonymization-benchmark`)
    qu'une URL Hugging Face complète, et renvoie l'identifiant que `load_dataset`
    attend.

    Exemples convertis vers `ildpil/text-anonymization-benchmark` :
      https://huggingface.co/datasets/ildpil/text-anonymization-benchmark
      https://huggingface.co/datasets/ildpil/text-anonymization-benchmark/tree/main
      huggingface.co/datasets/ildpil/text-anonymization-benchmark
    """
    name = (value or "").strip()
    # Retire le fragment/paramètres de requête éventuels.
    name = name.split("?", 1)[0].split("#", 1)[0]
    for marker in ("huggingface.co/datasets/", "hf.co/datasets/"):
        if marker in name:
            name = name.split(marker, 1)[1]
            break
    # Retire un préfixe de schéma résiduel et les segments de navigation du site.
    name = name.replace("https://", "").replace("http://", "")
    for suffix_marker in ("/tree/", "/blob/", "/resolve/", "/viewer"):
        if suffix_marker in name:
            name = name.split(suffix_marker, 1)[0]
            break
    return name.strip("/")


def load_docs_from_huggingface(
    dataset_name: str,
    text_column: str = "text",
    split: str = "train",
    config: str | None = None,
    max_docs: int | None = None,
) -> list[dict]:
    """Charge des documents depuis un dataset Hugging Face (téléchargé via `datasets`).

    `dataset_name` peut être un identifiant de dépôt ou une URL complète (normalisée)."""
    from datasets import load_dataset

    dataset_name = normalize_hf_name(dataset_name)
    kwargs = {"split": split, "trust_remote_code": True}
    if config:
        dataset = load_dataset(dataset_name, config, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)

    if text_column not in dataset.column_names:
        raise ValueError(
            f"Colonne '{text_column}' absente du dataset. "
            f"Colonnes disponibles : {dataset.column_names}"
        )

    docs: list[dict] = []
    for row in dataset:
        text = str(row.get(text_column, "") or "")
        if not text.strip():
            continue
        meta = {
            k: str(v)[:256]
            for k, v in row.items()
            if k != text_column and v is not None
        }
        docs.append({"doc_id": f"doc_{len(docs):04d}", "text": text, "metadata": meta})
        if max_docs and len(docs) >= max_docs:
            break
    if not docs:
        raise ValueError(f"Aucun texte non vide dans la colonne '{text_column}' du dataset.")
    return docs


def load_docs_from_text(raw_text: str, separator: str = "\n\n") -> list[dict]:
    """Découpe un texte brut collé en documents selon `separator`."""
    parts = raw_text.split(separator) if separator else [raw_text]
    docs: list[dict] = []
    for part in parts:
        if part.strip():
            docs.append(
                {"doc_id": f"doc_{len(docs):04d}", "text": part.strip(), "metadata": {}}
            )
    if not docs:
        raise ValueError("Le texte fourni est vide.")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Découpage en chunks
# ══════════════════════════════════════════════════════════════════════════════

def chunk_documents(
    docs: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Découpe chaque document (RecursiveCharacterTextSplitter, comme le reste du
    projet). Retourne des dicts prêts pour l'indexation."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[dict] = []
    for doc in docs:
        text = doc["text"]
        if not text.strip():
            continue
        cursor = 0
        for j, piece in enumerate(splitter.split_text(text)):
            start = text.find(piece, cursor)
            if start < 0:
                start = cursor
            end = start + len(piece)
            cursor = max(end - chunk_overlap, 0)
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{j:04d}",
                    "doc_id": doc["doc_id"],
                    "chunk_index": j,
                    "text": piece,
                    "char_start": start,
                    "char_end": end,
                    "pii_entities": [],
                }
            )
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Base vectorielle ISOLÉE (mirror de ChromaStore, dossier + collection dédiés)
# ══════════════════════════════════════════════════════════════════════════════

class IsolatedChromaStore:
    """ChromaDB persistante isolée. Même interface que vectorstore.ChromaStore
    (`.collection`, `.embedder`, `.query`, `.count`) pour être compatible avec
    B0 / NaiveRAG / HybridRAG, mais chemin et collection configurables afin de
    ne jamais toucher la base de benchmark."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        embedder=None,
    ):
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = str(persist_dir)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        if embedder is not None:
            self.embedder = embedder
        else:
            from embeddings.embedder import Embedder
            self.embedder = Embedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Vide la collection (repart de zéro), sans toucher les autres collections."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: list[dict], batch_size: int = 100) -> int:
        """Indexe les chunks (embeddings + métadonnées). Retourne le nombre total
        de chunks dans la collection après indexation."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            embeddings = self.embedder.embed_texts(texts, batch_size=batch_size).tolist()
            metadatas = [
                {
                    "doc_id": c["doc_id"],
                    "chunk_index": c.get("chunk_index", 0),
                    "char_start": c.get("char_start", 0),
                    "char_end": c.get("char_end", 0),
                    "n_pii": len(c.get("pii_entities", [])),
                    "pii_entities": json.dumps(c.get("pii_entities", [])),
                }
                for c in batch
            ]
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        return self.collection.count()

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed_single(query_text).tolist()
        n_results = min(top_k * 3, max(self.collection.count(), 1))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        chunks, seen = [], set()
        for j in range(len(results["ids"][0])):
            doc_id = results["metadatas"][0][j]["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            chunks.append(
                {
                    "chunk_id": results["ids"][0][j],
                    "text": results["documents"][0][j],
                    "similarity_score": 1 - results["distances"][0][j],
                    "doc_id": doc_id,
                }
            )
            if len(chunks) >= top_k:
                break
        return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Analyse B0 (countermeasure v6) — domaine + catégories + taxonomie + combos
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class B0Analysis:
    domain: str
    domain_confidence: float
    domain_source: str
    categories: list[str]
    taxonomy: dict[str, list[str]]
    category_hints: dict[str, list[str]]
    learned_types: list[str]
    risky_combos: list[list[str]]
    used_fallback: bool = False
    raw: dict = field(default_factory=dict)


def run_b0_analysis(store) -> B0Analysis:
    """Lance B0 exactement comme la countermeasure v6 : construit
    CPBNaiveRAGV6 autour du store (ce qui exécute le bootstrap B0 et la
    découverte des combinaisons ré-identifiantes), puis lit son résultat.

    NaiveRAG suffit ici (aucun retrieval n'est exercé — seuls B0 + combos sont
    calculés). Le LLM local (Ollama/Llama) doit être disponible pour les étapes
    LLM de B0 ; sinon le bootstrap retombe proprement sur ses replis."""
    from rag.naive_rag import NaiveRAG
    from llms.llama_llm import LlamaLLM
    from countermeasure_v6.cpb_naive_rag_v6 import CPBNaiveRAGV6

    llm = LlamaLLM()
    naive = NaiveRAG(store=store, llm=llm)
    combo = CPBNaiveRAGV6(naive_rag=naive)

    br = combo.bootstrap_result
    category_hints = {k: sorted(str(t) for t in v) for k, v in (br.category_hints or {}).items()}
    return B0Analysis(
        domain=br.domain,
        domain_confidence=float(br.domain_confidence),
        domain_source=br.domain_source,
        categories=list(br.dynamic_categories or []),
        taxonomy={k: list(v) for k, v in (br.dynamic_taxonomy or {}).items()},
        category_hints=category_hints,
        learned_types=sorted(str(t) for t in (br.learned_types or set())),
        risky_combos=[sorted(c) for c in (combo.risky_combos or [])],
        used_fallback=bool(br.used_fallback),
        raw={
            "domain": br.domain,
            "domain_confidence": float(br.domain_confidence),
            "domain_source": br.domain_source,
            "categories": list(br.dynamic_categories or []),
            "category_hints": category_hints,
            "taxonomy": {k: list(v) for k, v in (br.dynamic_taxonomy or {}).items()},
            "learned_types": sorted(str(t) for t in (br.learned_types or set())),
            "risky_combinations": [sorted(c) for c in (combo.risky_combos or [])],
            "used_fallback": bool(br.used_fallback),
        },
    )


def save_analysis(analysis: B0Analysis, persist_dir: str | Path) -> str:
    """Persiste l'analyse B0 (JSON) à côté de la base, pour traçabilité."""
    out = Path(persist_dir) / "b0_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(analysis.raw, f, ensure_ascii=False, indent=2)
    return str(out)
