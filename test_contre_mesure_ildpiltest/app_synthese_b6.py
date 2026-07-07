"""
app_synthese_b6.py — Interface Streamlit d'inspection de B6 (SAD detector) de CPB v4.

Pose UNE question et montre, étape par étape, ce que fait la contre-mesure sur la
réponse — en particulier le AVANT → APRÈS quand B6 décide de **synthétiser**
(réécrire) au lieu de masquer/bloquer :

    B6 decision ∈ {pass, synthesize, mask, block}
      - pass       : rien de sensible détecté       -> AVANT == APRÈS
      - synthesize : Phi-3 réécrit pour retirer le lien sensible (prose naturelle)
      - mask       : phrases sensibles -> [SENSITIVE_ATTRIBUTE_REDACTED]
      - block      : refus complet

N'appelle AUCUNE API externe : génération = Llama local (Ollama), B6 = Phi-3 local.
Ne modifie pas le code cœur : orchestre seulement les sous-étapes publiques de
CPBNaiveRAGV4 (retrieve -> generate -> sad_detector.detect -> response_guard.guard)
pour capturer la réponse brute (avant B6) ET la réponse B6 (après).

Lancement (sur la machine qui a la ChromaDB indexée + Ollama) :
    streamlit run test_contre_mesure_ildpiltest/app_synthese_b6.py
"""
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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from config import TOP_K

st.set_page_config(page_title="CPB v4 — Inspecteur B6 (synthèse)", layout="wide", page_icon="🔎")

# ── Couleurs par décision ─────────────────────────────────────────────────────
DECISION_STYLE = {
    "pass":                ("✅ PASS",        "#16a34a", "Rien de sensible détecté — réponse inchangée."),
    "synthesize":          ("✍️ SYNTHESIZE",  "#2563eb", "Phi-3 a réécrit la réponse pour retirer le lien sensible."),
    "mask":                ("▓ MASK",         "#d97706", "Phrases sensibles caviardées ([SENSITIVE_ATTRIBUTE_REDACTED])."),
    "block":               ("⛔ BLOCK",        "#dc2626", "Refus complet — trop sensible pour être divulgué."),
    "direct_suppression":  ("🚫 SUPPRESSION",  "#dc2626", "Requête supprimée en amont (B2 budget gate) — jamais générée."),
}

EXAMPLES = [
    "What health issues did Mr Omojudi face?",
    "Tell me about the applicant's medical condition in this case.",
    "What was the political affiliation of the person mentioned?",
]


# ── Chargement système (une seule fois, mis en cache) ─────────────────────────
@st.cache_resource(show_spinner="Chargement CPB v4 (B0 bootstrap + Llama, 1re fois seulement)…")
def load_system(retrieval_mode: str):
    from test_contre_mesure_ildpiltest._store import IldpilTestStore
    from test_contre_mesure_ildpiltest.config import CHROMA_DIR, COLLECTION_NAME
    from llms.llama_llm import LlamaLLM
    from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
    from countermeasure_v4.cpb_ablation import AblationConfig

    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    llm = LlamaLLM()
    if retrieval_mode == "hybrid":
        from rag.hybrid_rag import HybridRAG
        retriever = HybridRAG(store=store, llm=llm)
    else:
        from rag.naive_rag import NaiveRAG
        retriever = NaiveRAG(store=store, llm=llm)
    # Full pipeline (toutes les couches actives) pour que B6 puisse synthétiser.
    cpb = CPBNaiveRAGV4(naive_rag=retriever, ablation=AblationConfig(name="full_pipeline"))
    return cpb


# ── Une inspection : capture AVANT (brut) et APRÈS (B6), puis B7 ──────────────
def inspect(cpb, query: str, top_k: int) -> dict:
    retrieval = cpb.retrieve(query, top_k=top_k)
    masked_query = retrieval.get("masked_query", query)

    if retrieval["decision"] == "direct_suppression":
        return {
            "decision": "direct_suppression",
            "masked_query": masked_query,
            "chunks": [],
            "raw": None, "after_b6": None, "final": None,
            "sad": None,
        }

    chunks = retrieval["chunks"]
    llm_response = cpb.generate(masked_query, chunks)
    raw = llm_response.response                                    # AVANT B6

    sad = cpb.sad_detector.detect(query=masked_query, chunks=chunks, response=raw)
    after_b6 = sad.response                                        # APRÈS B6

    def reask():
        strengthened = (
            masked_query
            + "\n\nPrivacy instruction: answer only from the masked context. "
            + "Do not reveal names, identifiers, locations, dates, or any raw personal data. "
            + "Keep placeholders exactly as provided."
        )
        return cpb.generate(strengthened, chunks).response

    guarded = cpb.response_guard.guard(response=sad.response, reask_callback=reask)

    return {
        "decision": sad.decision,
        "masked_query": masked_query,
        "chunks": chunks,
        "raw": raw,
        "after_b6": after_b6,
        "final": guarded.response,
        "sad": sad,
        "b7_decision": guarded.decision,
    }


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔎 CPB v4 — Inspecteur B6")
st.caption("Voir le AVANT → APRÈS de la contre-mesure (synthèse / masquage / blocage) sur une question")

with st.sidebar:
    st.header("Réglages")
    retrieval_mode = st.radio("Retrieval", ["hybrid", "dense"], index=0,
                              help="hybrid = dense ChromaDB + BM25 (RRF) ; dense = vecteur seul")
    top_k = st.slider("top_k (chunks récupérés)", 1, 10, TOP_K)
    st.markdown("---")
    st.markdown("**Exemples** (cliquer pour remplir) :")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

query = st.text_input("Ta question :", key="query", placeholder="What health issues did Mr Omojudi face?")
go = st.button("▶️ Analyser", type="primary")

if go and query.strip():
    cpb = load_system(retrieval_mode)
    with st.spinner("Génération Llama + analyse B6 (Phi-3)…"):
        res = inspect(cpb, query.strip(), top_k)

    label, color, desc = DECISION_STYLE.get(res["decision"], (res["decision"], "#666", ""))
    st.markdown(
        f"<div style='padding:10px 16px;border-radius:8px;background:{color}22;"
        f"border-left:5px solid {color};font-size:1.1rem'>"
        f"<b style='color:{color}'>Décision B6 : {label}</b><br>"
        f"<span style='opacity:.85'>{desc}</span></div>",
        unsafe_allow_html=True,
    )

    if res["decision"] == "direct_suppression":
        st.warning("La requête a été supprimée par B2 (budget gate) avant toute génération — "
                   "il n'y a donc pas de réponse à comparer.")
    else:
        sad = res["sad"]
        # ── AVANT / APRÈS côte à côte ──
        st.subheader("Réponse : avant → après B6")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟠 AVANT B6** (réponse brute du LLM)")
            st.info(res["raw"] or "_(vide)_")
        with c2:
            changed = res["after_b6"] != res["raw"]
            st.markdown(f"**{'🔵' if changed else '⚪'} APRÈS B6** "
                        f"({'modifiée' if changed else 'inchangée'})")
            (st.success if changed else st.info)(res["after_b6"] or "_(vide)_")

        if res["final"] != res["after_b6"]:
            st.markdown("**Après B7 (response guard)** — nettoyage PII résiduelle :")
            st.warning(res["final"])

        # ── Détails de la décision B6 ──
        with st.expander("🧠 Détails de la décision B6", expanded=res["decision"] != "pass"):
            st.write(f"**Catégories sensibles détectées :** "
                     f"{', '.join(sad.attribute_categories) or '— aucune —'}")
            st.write(f"**Similarité max (SBERT) :** {sad.max_similarity:.3f}")
            st.write(f"**Confiance :** {sad.confidence:.2f}")
            st.write(f"**Filtre déclencheur :** F{sad.filter_triggered}")
            st.write(f"**Raisonnement :** {sad.reasoning}")

        # ── Contexte récupéré ──
        with st.expander(f"📄 Chunks récupérés ({len(res['chunks'])}) — après masquage B3/B4"):
            for i, ch in enumerate(res["chunks"]):
                sim = ch.get("similarity_score")
                sim_s = f"{sim:.3f}" if isinstance(sim, (int, float)) else "n/a"
                st.markdown(f"**[{i}]** sim={sim_s} · doc_id=`{ch.get('doc_id')}` · "
                            f"sensibilité=`{ch.get('sensitivity', '?')}`")
                st.caption((ch.get("text", "") or "")[:500])

        if res["masked_query"] != query.strip():
            st.caption(f"ℹ️ Requête masquée envoyée au LLM : {res['masked_query']}")

elif go:
    st.error("Écris une question d'abord.")
