# rag/hybrid_rag.py
"""
HybridRAG — retrieval hybride dense (ChromaDB, cosinus) + sparse (BM25 lexical),
fusionné par Reciprocal Rank Fusion (RRF).

Motivation : le dense seul floute les tokens rares (noms propres, numéros de
dossier) → il ramène le mauvais document quand la question nomme "Mr Gunnar
Beck" ou "64735/01". BM25 matche exactement ces tokens. RRF combine les deux
classements par RANG (pas par score), donc sans normaliser des échelles
incompatibles (cosinus 0-1 vs BM25 0-∞).

Drop-in de NaiveRAG : mêmes signatures retrieve()/generate()/run(), même format
de chunk en sortie → utilisable partout où NaiveRAG l'est, y compris comme
`naive_rag=` de CPBNaiveRAGV4.

Dépendance : rank_bm25  (pip install rank_bm25)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llms.base_llm import BaseLLM, LLMResponse
from config import TOP_K

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Tokenisation lexicale simple. "64735/01" → ["64735","01"] : la question
    se tokenise pareil, donc le match sur "64735" (token rare) fonctionne."""
    return _TOKEN_RE.findall((text or "").lower())


class HybridRAG:
    """RAG hybride dense + BM25, fusion RRF. Interface identique à NaiveRAG."""

    def __init__(
        self,
        store,
        llm: BaseLLM,
        candidate_k: int = 20,   # nb de candidats pris à CHAQUE retriever avant fusion
        rrf_k: int = 60,         # constante RRF (standard = 60)
        dedup: bool = True,      # True : 1 chunk par doc (fusion par doc_id).
                                 # False : plusieurs chunks/doc (fusion par chunk_id).
    ):
        self.store = store
        self.llm = llm
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dedup = dedup
        self._build_bm25()

    # ── BM25 index (construit une fois sur tous les chunks) ────────────────────

    def _build_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "HybridRAG nécessite rank_bm25 : pip install rank_bm25"
            ) from exc

        got = self.store.collection.get(include=["documents", "metadatas"])
        self._ids   = got.get("ids", []) or []
        self._texts = got.get("documents", []) or []
        self._metas = got.get("metadatas", []) or []
        if not self._texts:
            raise RuntimeError("HybridRAG : collection vide, rien à indexer pour BM25.")

        self._bm25 = BM25Okapi([_tokenize(t) for t in self._texts])

    # ── Construction d'un chunk au format store.query() ────────────────────────

    def _chunk_from_index(self, i: int) -> dict:
        meta = self._metas[i] or {}
        raw_pii = meta.get("pii_entities", "[]")
        try:
            pii = json.loads(raw_pii) if isinstance(raw_pii, str) else (raw_pii or [])
        except (json.JSONDecodeError, TypeError):
            pii = []
        return {
            "chunk_id":         self._ids[i],
            "text":             self._texts[i],
            "similarity_score": None,           # score dense inconnu pour un hit BM25
            "doc_id":           meta.get("doc_id"),
            "n_pii":            meta.get("n_pii", len(pii)),
            "pii_entities":     pii,
            "sensitivity":      meta.get("sensitivity", "NOT_CONFIDENTIAL"),
        }

    def _bm25_topk(self, query: str) -> list[dict]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order[:self.candidate_k]:
            if scores[i] <= 0:
                break                            # plus aucun token en commun
            ch = self._chunk_from_index(i)
            ch["bm25_score"] = round(float(scores[i]), 4)
            out.append(ch)
        return out

    # ── Fusion RRF (par doc_id, comme NaiveRAG renvoie des docs uniques) ────────

    def _rrf_fuse(self, dense: list[dict], sparse: list[dict], top_k: int) -> list[dict]:
        # dedup=True  : fusion par doc_id   → 1 chunk par document.
        # dedup=False : fusion par chunk_id → plusieurs chunks du même document
        #               survivent (le doc source apparaît en plusieurs morceaux).
        key_field = "doc_id" if self.dedup else "chunk_id"
        acc: dict = {}   # key -> {"score":float, "chunk":dict}
        for ranked in (dense, sparse):
            for rank, ch in enumerate(ranked):
                key = ch.get(key_field)
                if key is None:
                    continue
                entry = acc.setdefault(key, {"score": 0.0, "chunk": ch})
                entry["score"] += 1.0 / (self.rrf_k + rank + 1)
                # Préférer le chunk dense (il porte le cosinus) comme représentant.
                if ch.get("similarity_score") is not None and entry["chunk"].get("similarity_score") is None:
                    entry["chunk"] = ch

        fused = sorted(acc.values(), key=lambda e: e["score"], reverse=True)
        out = []
        for e in fused[:top_k]:
            ch = dict(e["chunk"])
            ch["rrf_score"] = round(e["score"], 6)
            out.append(ch)
        return out

    # ── Interface NaiveRAG ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        dense  = self.store.query(query, top_k=self.candidate_k)
        sparse = self._bm25_topk(query)
        return self._rrf_fuse(dense, sparse, top_k)

    def generate(self, query: str, chunks: list[dict]) -> LLMResponse:
        prompt = self.llm.build_rag_prompt(query, chunks)
        return self.llm.generate(prompt)

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        chunks = self.retrieve(query, top_k)
        result = self.generate(query, chunks)
        return {
            "query":             query,
            "chunks":            chunks,
            "response":          result.response,
            "architecture":      "hybrid_rag",
            "llm":               result.llm_name,
            "tokens_prompt":     result.tokens_prompt,
            "tokens_completion": result.tokens_completion,
            "tokens_total":      result.tokens_total,
            "cost_usd":          result.cost_usd,
        }
