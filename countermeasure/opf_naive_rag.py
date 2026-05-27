# countermeasure/opf_naive_rag.py
"""
OPFNaiveRAG — NaiveRAG + OpenAI Privacy Filter comme contre-mesure (post-génération).

Pipeline :
  1. Retrieve   — top-k chunks via NaiveRAG (cosinus, pas de filtrage)
  2. Generate   — LLM répond sur le contexte COMPLET (meilleure qualité)
  3. Redact     — OPF redacte les PII dans la RÉPONSE du LLM

Avantages vs redaction pré-génération :
  - LLM voit le contexte complet → réponses de meilleure qualité
  - OPF traite seulement la réponse courte (pas tous les chunks) → + rapide
  - Séparation claire : génération = LLM, filtrage sortie = OPF

Comparé à CPBNaiveRAG :
  - Pas de QueryRiskScorer, pas de BudgetGate, pas de SADDetector
  - Redaction basée sur un modèle NLP (1.5B params, 50M actifs)
  - Fonctionne 100% localement, aucune API externe
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from transformers import pipeline as hf_pipeline

from config import TOP_K
from rag.naive_rag import NaiveRAG


# ── Utilitaires OPF ──────────────────────────────────────────────────────────

def _redact_text(clf, text: str) -> tuple[str, list[dict]]:
    """
    Applique OPF sur un texte et retourne (texte_redacté, entités_détectées).
    Limite à 1500 caractères pour rester dans la fenêtre du modèle.
    """
    if not text or not text.strip():
        return text or "", []

    # OPF supporte jusqu'à 128k tokens, mais on tronque à 1500 chars par sécurité
    text_input = text[:1500]
    entities   = clf(text_input)

    detected = [
        {
            "entity": e["entity_group"],
            "word":   e["word"],
            "score":  round(e["score"], 3),
            "start":  e["start"],
            "end":    e["end"],
        }
        for e in entities
    ]

    # Redaction : trier par position décroissante pour ne pas décaler les indices
    redacted = text_input
    for ent in sorted(detected, key=lambda x: x["start"], reverse=True):
        label    = ent["entity"]
        redacted = redacted[: ent["start"]] + f"[{label}]" + redacted[ent["end"]:]

    # Si le texte original était plus long que 1500 chars, on garde le reste intact
    if len(text) > 1500:
        redacted = redacted + text[1500:]

    return redacted, detected


# ── Classe principale ─────────────────────────────────────────────────────────

class OPFNaiveRAG:
    """
    Contre-mesure OPF autour de NaiveRAG — redaction POST-génération.

    Pipeline :
      1  Retrieval NaiveRAG (top-k cosinus) — chunks bruts
      2  Génération LLM sur contexte complet (meilleure qualité)
      3  OPF redacte les PII dans la RÉPONSE du LLM

    Le LLM voit le contexte complet, OPF filtre uniquement ce qui sort.
    """

    MODEL_ID = "openai/privacy-filter"

    def __init__(
        self,
        naive_rag: NaiveRAG,
        architecture_name: str = "opf_naive_rag",
        device: str | None = None,       # None = auto (CPU ou GPU)
    ):
        self.naive_rag         = naive_rag
        self.store             = naive_rag.store
        self.llm               = naive_rag.llm
        self.architecture_name = architecture_name

        print(f"⏳ Chargement OPF ({self.MODEL_ID})...")
        t0 = time.time()
        kwargs = {"task": "token-classification",
                  "model": self.MODEL_ID,
                  "aggregation_strategy": "simple",
                  "trust_remote_code": True}    # requis : architecture custom openai_privacy_filter
        if device is not None:
            kwargs["device"] = device

        self._clf = hf_pipeline(**kwargs)
        print(f"✅ OPF chargé en {time.time() - t0:.1f}s")

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve les chunks bruts (pas de redaction à cette étape)."""
        return self.naive_rag.retrieve(query, top_k=top_k)

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: list[dict]):
        """LLM génère sur le contexte complet (chunks bruts)."""
        return self.naive_rag.generate(query, chunks)

    # ── Run complet ───────────────────────────────────────────────────────────

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Pipeline complète OPFNaiveRAG :
          retrieve → generate (LLM sur chunks bruts) → redact réponse (OPF)

        Le LLM voit le contexte complet → meilleure qualité de réponse.
        OPF traite uniquement la réponse courte → plus rapide.
        """
        # ── 1. Retrieve (chunks bruts, pas de filtrage)
        raw_chunks = self.retrieve(query, top_k=top_k)

        # ── 2. Generate (LLM voit le contexte complet)
        llm_response = self.generate(query, raw_chunks)
        raw_response = llm_response.response

        # ── 3. Redact : OPF filtre les PII dans la RÉPONSE
        t_opf = time.time()
        redacted_response, entities = _redact_text(self._clf, raw_response)
        opf_latency = time.time() - t_opf

        return {
            # ── Résultat principal
            "query":              query,
            "response":           redacted_response,   # réponse finale (sans PII)
            "raw_response":       raw_response,        # réponse brute du LLM (avant OPF)
            "architecture":       self.architecture_name,
            "llm":                llm_response.llm_name,
            # ── Chunks (bruts = envoyés au LLM = utilisés pour mesure PII GT)
            "chunks":             raw_chunks,
            "raw_chunks":         raw_chunks,
            # ── Métriques OPF (sur la réponse)
            "opf_entities_total":  len(entities),
            "opf_entities":        entities,
            "opf_latency_s":       round(opf_latency, 3),
            # ── Tokens / coût
            "tokens_prompt":      llm_response.tokens_prompt,
            "tokens_completion":  llm_response.tokens_completion,
            "tokens_total":       llm_response.tokens_total,
            "cost_usd":           llm_response.cost_usd,
        }
