# rag/hhr_rag.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
from vectorstore.chroma_store import ChromaStore
from llms.base_llm import BaseLLM, LLMResponse
from config import TOP_K


class HHRRAG:
    """
    Hybrid Hierarchical Retrieval — Arivazhagan et al. 2023.
    
    Adapté pour ildpil (pas de structure titre/abstract) :
    
    Étape 1 — BM25 (sparse) sur tous les chunks
              → identifie les top-kd documents pertinents
    
    Étape 2 — Bi-encoder (dense) sur les chunks
              uniquement dans les documents sélectionnés
              → reranking final top-k passages
    
    Configuration : Sparse+Dense (meilleur compromis selon Table 1)
    """

    # Paramètres hiérarchiques selon l'article
    KD = 10   # top-kd documents récupérés en étape 1
    KP = TOP_K  # top-kp passages récupérés en étape 2

    def __init__(self, store: ChromaStore, llm: BaseLLM):
        self.store  = store
        self.llm    = llm
        self._bm25  = None
        self._corpus = None   # liste de chunks indexés pour BM25

    # ============================================================
    #  BM25 INDEX — construit une seule fois à la demande
    # ============================================================
    def _build_bm25_index(self):
        """
        Construit l'index BM25 depuis tous les chunks ChromaDB.
        Appelé une seule fois puis mis en cache.
        """
        if self._bm25 is not None:
            return

        print("📥 Construction index BM25...")

        # Récupérer tous les chunks depuis ChromaDB
        results = self.store.collection.get(
            include=["documents", "metadatas"],
        )

        self._corpus = []
        for i in range(len(results["ids"])):
            self._corpus.append({
                "chunk_id": results["ids"][i],
                "text":     results["documents"][i],
                "doc_id":   results["metadatas"][i]["doc_id"],
                "n_pii":    results["metadatas"][i]["n_pii"],
                "pii_entities": results["metadatas"][i].get("pii_entities", "[]"),
            })

        # Tokenisation simple pour BM25
        tokenized = [chunk["text"].lower().split() for chunk in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        print(f"✅ Index BM25 construit — {len(self._corpus)} chunks")

    # ============================================================
    #  ÉTAPE 1 — BM25 : top-kd documents
    # ============================================================
    def _retrieve_documents_bm25(self, query: str) -> list[str]:
        """
        Étape 1 — Sparse retrieval BM25.
        Retourne les doc_ids des KD documents les plus pertinents.
        """
        self._build_bm25_index()

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Associer scores aux chunks
        scored_chunks = list(zip(scores, self._corpus))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Collecter les doc_ids uniques des KD meilleurs chunks
        seen_doc_ids = set()
        top_doc_ids  = []

        for score, chunk in scored_chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                top_doc_ids.append(doc_id)
            if len(top_doc_ids) >= self.KD:
                break

        return top_doc_ids

    # ============================================================
    #  ÉTAPE 2 — Dense : reranking dans les documents sélectionnés
    # ============================================================
    def _retrieve_passages_dense(
        self, query: str, doc_ids: list[str]
    ) -> list[dict]:
        """
        Étape 2 — Dense retrieval via bi-encoder (ChromaDB).
        Reranke uniquement les passages des documents sélectionnés.
        Correspond au passage retriever on-the-fly de l'article (Appendix A).
        """
        import json

        query_embedding = self.store.embedder.embed_single(query).tolist()

        # Récupérer tous les passages des documents sélectionnés
        results = self.store.collection.get(
            where={"doc_id": {"$in": doc_ids}},
            include=["documents", "metadatas", "embeddings"],
        )

        if not results["ids"]:
            return []

        # Calculer similarité cosinus entre query et chaque passage
        import numpy as np

        query_vec = np.array(query_embedding)
        passages  = []

        for i in range(len(results["ids"])):
            passage_vec = np.array(results["embeddings"][i])
            # Similarité cosinus — vecteurs déjà normalisés
            similarity = float(np.dot(query_vec, passage_vec))

            passages.append({
                "chunk_id":       results["ids"][i],
                "text":           results["documents"][i],
                "similarity_score": similarity,
                "doc_id":         results["metadatas"][i]["doc_id"],
                "n_pii":          results["metadatas"][i]["n_pii"],
                "pii_entities":   json.loads(
                    results["metadatas"][i].get("pii_entities", "[]")
                ),
            })

        # Trier par similarité décroissante
        passages.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Dédupliquer par doc_id — garder le meilleur chunk par document
        seen_doc_ids = set()
        top_passages = []

        for p in passages:
            if p["doc_id"] not in seen_doc_ids:
                seen_doc_ids.add(p["doc_id"])
                top_passages.append(p)
            if len(top_passages) >= self.KP:
                break

        return top_passages

    # ============================================================
    #  RETRIEVE — pipeline hiérarchique complet
    # ============================================================
    def retrieve(self, query: str, top_k: int = None) -> dict:
        """
        Pipeline HHR complet : BM25 → Dense reranking.
        Retourne les chunks + métadonnées des deux étapes.
        """
        # Étape 1 — BM25
        top_doc_ids = self._retrieve_documents_bm25(query)

        # Étape 2 — Dense reranking
        top_passages = self._retrieve_passages_dense(query, top_doc_ids)

        return {
            "chunks":    top_passages,
            "doc_ids_stage1": top_doc_ids,   # pour logging
            "n_docs_stage1":  len(top_doc_ids),
        }

    # ============================================================
    #  GENERATE
    # ============================================================
    def generate(self, query: str, chunks: list[dict]) -> LLMResponse:
        prompt = self.llm.build_rag_prompt(query, chunks)
        return self.llm.generate(prompt)

    # ============================================================
    #  RUN — pipeline complet
    # ============================================================
    def run(self, query: str) -> dict:
        """
        Pipeline complète HHR : retrieve + generate.
        """
        retrieval = self.retrieve(query)
        chunks    = retrieval["chunks"]
        result    = self.generate(query, chunks)

        return {
            "query":            query,
            "chunks":           chunks,
            "response":         result.response,
            "architecture":     "hhr_rag",
            "llm":              result.llm_name,
            # HHR spécifique
            "n_docs_stage1":    retrieval["n_docs_stage1"],
            "doc_ids_stage1":   retrieval["doc_ids_stage1"],
            # tokens
            "tokens_prompt":    result.tokens_prompt,
            "tokens_completion": result.tokens_completion,
            "tokens_total":     result.tokens_total,
            # coût
            "cost_usd":         result.cost_usd,
        }

