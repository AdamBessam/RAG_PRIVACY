"""
Interface chat (style ChatGPT) sur le dataset financier protégé par CPB NaiveRAG v2.

Usage :
    streamlit run benchmark_financial/chat_app.py
    streamlit run benchmark_financial/chat_app.py -- --llm claude-haiku
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from benchmark_financial.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from benchmark_financial._store import FinancialStore
from contre_mesure_nv.cpb_naive_rag import CPBNaiveRAG
from rag.naive_rag import NaiveRAG


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial RAG + CPB",
    page_icon="🔒",
    layout="wide",
)


# ── LLM builder ───────────────────────────────────────────────────────────────

def build_llm(llm_name: str):
    if llm_name == "llama":
        from llms.llama_llm import LlamaLLM
        return LlamaLLM()
    if llm_name == "mistral":
        from llms.mistral_llm import MistralLLM
        return MistralLLM()
    if llm_name == "claude-haiku":
        from llms.claude_haiku_llm import ClaudeHaikuLLM
        return ClaudeHaikuLLM()
    if llm_name == "gpt4o-mini":
        from llms.gpt4o_mini_llm import GPT4oMiniLLM
        return GPT4oMiniLLM()
    raise ValueError(f"LLM inconnu : {llm_name}")


@st.cache_resource(show_spinner="Initialisation du store ChromaDB et de CPB...")
def load_cpb(llm_name: str) -> CPBNaiveRAG:
    store = FinancialStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    if store.count() == 0:
        st.error(
            "Collection vide. "
            "Lancez d'abord : python benchmark_financial/01_index.py"
        )
        st.stop()
    llm       = build_llm(llm_name)
    naive_rag = NaiveRAG(store=store, llm=llm)
    return CPBNaiveRAG(
        naive_rag=naive_rag,
        architecture_name="cpb_naive_rag_financial_chat",
    )


# ── Analyse des fuites PII ────────────────────────────────────────────────────

def analyze_pii_leakage(response: str, raw_chunks: list[dict]) -> dict:
    """
    Compare les entités PII des raw_chunks (GT) avec la réponse finale.
    Retourne deux listes : leaked (fuité) et protected (filtré par CPB).
    """
    leaked: list[dict] = []
    protected: list[dict] = []
    seen: set[str] = set()
    response_lower = response.lower()

    for chunk in raw_chunks:
        for ent in chunk.get("pii_entities", []):
            text  = ent.get("text", "").strip()
            etype = (
                ent.get("type")
                or ent.get("entity_type")
                or ent.get("label", "PII")
            )
            if len(text) <= 2 or text.lower() in seen:
                continue
            seen.add(text.lower())
            entry = {"text": text, "type": str(etype).upper()}
            if text.lower() in response_lower:
                leaked.append(entry)
            else:
                protected.append(entry)

    return {"leaked": leaked, "protected": protected}


# ── Couleur de badge selon la décision CPB ────────────────────────────────────

DECISION_COLOR = {
    "direct_suppression":   "🔴",
    "all_chunks_suppressed": "🔴",
    "reask":                "🟠",
    "mask":                 "🟡",
    "retrieval_masked":     "🟡",
    "pass":                 "🟢",
    "unknown":              "⚪",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    llm_choice = st.selectbox(
        "Modèle LLM",
        ["llama", "mistral", "claude-haiku", "gpt4o-mini"],
        index=0,
        help="Sélectionnez le LLM utilisé pour générer les réponses.",
    )

    st.divider()
    st.markdown("### 🔒 CPB NaiveRAG v2")
    st.markdown(
        "- **QueryRiskScorer** adaptatif (context targets dynamiques)\n"
        "- **SAD Detector** avec taxonomie enrichie depuis le store\n"
        "- **Presidio** PII Analyzer + Anonymizer\n"
        "- **BudgetGate** par chunk\n"
        "- **ResponseGuard** sur la réponse finale"
    )

    st.divider()
    st.markdown("**Légende des décisions :**")
    for decision, badge in DECISION_COLOR.items():
        st.markdown(f"{badge} `{decision}`")

    st.divider()
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Initialisation ────────────────────────────────────────────────────────────

cpb = load_cpb(llm_choice)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "blocked": 0, "pii_leaked": 0, "pii_protected": 0}


# ── En-tête ───────────────────────────────────────────────────────────────────

st.title("💬 Financial RAG — Protégé par CPB v2")
st.caption(
    "Posez vos questions sur le dataset financier. "
    "La contre-mesure CPB filtre automatiquement les informations sensibles."
)

# Stats globales de la session
stats = st.session_state.stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("Questions posées",  stats.get("total", 0))
c2.metric("Requêtes bloquées", stats.get("blocked", 0))
c3.metric("🔴 PII divulguées", stats.get("pii_leaked", 0))
c4.metric("🟢 PII protégées",  stats.get("pii_protected", 0))

st.divider()


# ── Helper : affiche les détails CPB d'un message ────────────────────────────

def render_cpb_details(meta: dict) -> None:
    with st.expander("🔍 Détails CPB", expanded=False):
        col1, col2, col3 = st.columns(3)
        badge = DECISION_COLOR.get(meta["decision"], "⚪")
        col1.metric("Score de risque", f"{meta['risk']:.3f}")
        col2.metric("Décision", f"{badge} {meta['decision']}")
        col3.metric("Latence", f"{meta['latency']:.2f} s")

        signals = meta.get("signals", {})
        if signals:
            st.markdown("**Décomposition du score de risque :**")
            sig_cols = st.columns(len(signals))
            labels = {
                "s1_ner":        "NER",
                "s2_extractive": "Extractif",
                "s3_jailbreak":  "Jailbreak",
                "s4_session":    "Session",
                "s5_semantic":   "Sémantique",
            }
            for i, (k, v) in enumerate(signals.items()):
                sig_cols[i].metric(labels.get(k, k), f"{v:.3f}")

        if meta.get("sad"):
            st.warning(
                f"⚠️ SAD détecté — catégories : {', '.join(meta['sad_categories'])}"
                f" (confiance : {meta['sad_confidence']:.2f})"
            )

        if meta.get("masked_query") and meta["masked_query"] != meta["query"]:
            st.info(f"🔏 Query anonymisée : `{meta['masked_query']}`")

        n_chunks = meta.get("n_chunks", 0)
        st.markdown(f"**Chunks récupérés :** {n_chunks}")

        # ── Analyse PII ───────────────────────────────────────────────────────
        leaked    = meta.get("pii_leaked", [])
        protected = meta.get("pii_protected", [])

        if leaked or protected:
            st.markdown("---")
            st.markdown("**🔎 Analyse des PII dans les chunks récupérés :**")

            col_l, col_p = st.columns(2)

            with col_l:
                st.markdown(f"🔴 **PII divulguées : {len(leaked)}**")
                if leaked:
                    for ent in leaked:
                        st.markdown(
                            f"<span style='background:#ffcccc;padding:2px 6px;"
                            f"border-radius:4px;font-size:0.85em'>"
                            f"<b>{ent['type']}</b> — {ent['text']}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("*Aucune fuite détectée*")

            with col_p:
                st.markdown(f"🟢 **PII protégées : {len(protected)}**")
                if protected:
                    for ent in protected:
                        st.markdown(
                            f"<span style='background:#ccffcc;padding:2px 6px;"
                            f"border-radius:4px;font-size:0.85em'>"
                            f"<b>{ent['type']}</b> — {ent['text']}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("*Aucune PII dans les chunks*")


# ── Rejouer l'historique ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            blocked = msg.get("blocked", False)
            if blocked:
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if "meta" in msg:
                render_cpb_details(msg["meta"])
        else:
            st.markdown(msg["content"])


# ── Zone de saisie ────────────────────────────────────────────────────────────

if prompt := st.chat_input("Posez votre question sur le dataset financier..."):

    # Message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Appel CPB + affichage de la réponse
    with st.chat_message("assistant"):
        with st.spinner("CPB en cours de traitement..."):
            t0  = time.time()
            out = cpb.run(prompt, top_k=TOP_K)
            latency = round(time.time() - t0, 2)

        response    = out.get("response", "")
        decision    = out.get("cpb_response_guard_decision",
                              out.get("cpb_sad_decision", "unknown"))
        risk        = float(out.get("cpb_query_risk", 0.0))
        signals     = out.get("cpb_query_risk_signals", {})
        sad         = bool(out.get("cpb_sad_detected", False))
        sad_cats    = out.get("cpb_sad_categories", [])
        sad_conf    = float(out.get("cpb_sad_confidence", 0.0))
        masked_q    = out.get("cpb_masked_query", prompt)
        raw_chunks  = out.get("raw_chunks", [])

        blocked = decision in ("direct_suppression", "all_chunks_suppressed")

        if blocked:
            st.error(response)
        else:
            st.markdown(response)

        pii_analysis = analyze_pii_leakage(response, raw_chunks)

        meta = {
            "risk":           risk,
            "signals":        signals,
            "decision":       decision,
            "latency":        latency,
            "sad":            sad,
            "sad_categories": sad_cats,
            "sad_confidence": sad_conf,
            "query":          prompt,
            "masked_query":   masked_q,
            "n_chunks":       len(raw_chunks),
            "pii_leaked":     pii_analysis["leaked"],
            "pii_protected":  pii_analysis["protected"],
        }
        render_cpb_details(meta)

    # Mise à jour des stats de session
    st.session_state.stats["total"] += 1
    if blocked:
        st.session_state.stats["blocked"] += 1
    st.session_state.stats["pii_leaked"]    += len(pii_analysis["leaked"])
    st.session_state.stats["pii_protected"] += len(pii_analysis["protected"])

    # Sauvegarde dans l'historique
    st.session_state.messages.append({
        "role":    "assistant",
        "content": response,
        "blocked": blocked,
        "meta":    meta,
    })

    st.rerun()
