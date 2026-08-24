"""
Interface 2 (Streamlit, ISOLÉE) — Chat façon ChatGPT protégé par CPS.

Lancer :   venv\\Scripts\\python.exe interface\\run.py chat

Elle interroge la base vectorielle remplie par l'interface 1 (ingestion), via :
  • retrieval HYBRIDE (dense ChromaDB + BM25, fusion RRF)  — HybridRAG
  • contre-mesure CPS : requête et chunks BRUTS (jamais masqués) envoyés au
    LLM ; seule la RÉPONSE générée est ensuite masquée sélectivement (poids +
    hints de domaine + combinaisons ré-identifiantes), puis vérifiée par le
    détecteur SAD (B6, dernier filet — pas de brique B7) — CPBNaiveRAGV6

Fonctionnalités chat :
  • sessions de conversation (barre latérale, comme ChatGPT), historique par session
  • chaque session porte son propre session_id CPS (scoring de risque multi-tour)
  • sous chaque réponse, un volet « Détails de confidentialité » (nombre de
    remplacements PII dans la réponse, décision SAD).

Cette interface ne modifie aucun fichier existant.
"""

import sys
from pathlib import Path

# torch d'abord (voir interface/run.py) — sécurité si lancé hors du lanceur.
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from uuid import uuid4

import streamlit as st

from config import TOP_K
from interface.ingestion import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR, IsolatedChromaStore

st.set_page_config(page_title="Chat CPS", layout="centered")


# ══════════════════════════════════════════════════════════════════════════════
# Habillage épuré — thème clair simple, sans noir (fond blanc/gris très clair)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
      :root {
        --bg: #ffffff;             /* fond principal */
        --sidebar-bg: #f7f7f8;     /* sidebar, gris très clair */
        --panel: #fafafa;          /* panneaux (expanders, code) */
        --bubble-user: #f0f0f1;    /* bulle utilisateur / avatar assistant */
        --text: #1f1f1f;           /* texte principal, pas de noir pur */
        --muted: #6e6e80;          /* texte secondaire */
        --line: #e4e4e6;           /* bordures */
        --line-soft: rgba(0,0,0,0.06);
        --hover: rgba(0,0,0,0.045);
        --accent: #1f1f1f;         /* bouton d'envoi */
        --accent-contrast: #ffffff;
        --font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Roboto, Helvetica, Arial, sans-serif;
      }

      html { font-size: 15px; }
      html, body, .stApp { background: var(--bg) !important; font-family: var(--font); }
      .block-container { max-width: 44rem; padding-top: 1.5rem; padding-bottom: 7rem; }
      h1 { font-size: 1.2rem; font-weight: 600; letter-spacing: -0.01em; color: var(--text); }
      p, label, span, li, [data-testid="stMarkdownContainer"] { color: var(--text); line-height: 1.6; }
      .app-sub { color: var(--muted); font-size: 0.8rem; margin-top: -0.4rem; }
      hr { border-color: var(--line-soft); }

      /* Chrome Streamlit superflu masqué (menu réglages, bouton Deploy, footer) —
         le contrôle d'ouverture/fermeture de la sidebar (mobile) reste visible. */
      [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], footer { display: none; }
      [data-testid="stHeader"] { background: var(--bg); }
      [data-testid="stExpandSidebarButton"] svg,
      [data-testid="stSidebarCollapseButton"] svg { color: var(--text) !important; }

      /* ── Sidebar (~260px, façon ChatGPT) ─────────────────────────────────── */
      [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--line-soft);
        transition: width 0.2s ease, min-width 0.2s ease;
      }
      [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 260px !important; width: 260px !important;
      }
      [data-testid="stSidebarContent"] {
        padding: 0.75rem 0.6rem;
        scrollbar-width: thin; scrollbar-color: var(--line) transparent;
      }
      [data-testid="stSidebarContent"]::-webkit-scrollbar { width: 6px; }
      [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb { background: var(--line); border-radius: 3px; }
      [data-testid="stSidebar"] h3 {
        color: var(--muted); font-size: 0.72rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.04em; padding: 0.5rem 0.4rem 0.3rem;
      }

      /* Items de conversation + bouton "nouvelle conversation" : ghost buttons */
      [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px !important; border: 1px solid transparent !important;
        background: transparent !important; color: var(--text) !important;
        font-weight: 400; font-size: 0.82rem; text-align: left; justify-content: flex-start;
        padding: 0.45rem 0.6rem; transition: background-color 0.15s ease;
      }
      [data-testid="stSidebar"] .stButton > button:hover { background: var(--hover) !important; }
      [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        border: 1px solid var(--line) !important; background: #ffffff !important;
        font-weight: 500; margin-bottom: 0.5rem;
      }
      [data-testid="stSidebar"] .stButton > button[kind="primary"] p::before { content: "＋  "; }
      [data-testid="stSidebar"] .stButton > button[kind="secondary"] p::before { content: "💬  "; opacity: 0.6; }

      [data-testid="stSidebar"] [data-testid="stTextInput"] input,
      [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] {
        background: #ffffff !important; color: var(--text) !important;
        border: 1px solid var(--line) !important; border-radius: 8px !important;
      }
      [data-testid="stSidebar"] [data-testid="stSliderTickBar"] { color: var(--muted) !important; }

      /* ── Bulles de messages : full-width assistant, bulle discrète pour l'utilisateur ── */
      [data-testid="stChatMessage"] {
        background: transparent; border: none; padding: 0.6rem 0; gap: 0.7rem;
        animation: cpb-fadein 0.25s ease-out; font-size: 0.92rem;
      }
      [data-testid="stChatMessageAvatarUser"] { display: none; }
      [data-testid="stChatMessageAvatarAssistant"] {
        width: 24px !important; height: 24px !important; border-radius: 50% !important;
        background: var(--bubble-user) !important; border: 1px solid var(--line-soft);
      }
      [data-testid="stChatMessageAvatarAssistant"] svg { color: var(--text) !important; fill: currentColor !important; }
      .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) { justify-content: flex-end; }
      .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: var(--bubble-user); border-radius: 18px; padding: 0.5rem 0.85rem; max-width: 70%;
      }
      .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background: transparent; padding: 0.15rem 0; width: 100%;
      }
      @keyframes cpb-fadein {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
      }

      /* ── Zone de saisie : barre arrondie, ombre légère, bouton rond ──────── */
      /* stBottom contient un div interne (sans data-testid, fond clair par défaut de
         Streamlit) qui déborde largement de la barre de saisie centrée — c'est lui qui
         produit les bandes blanches de part et d'autre de l'input tout en bas de page. */
      [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background: var(--bg) !important; }
      [data-testid="stBottom"] > div { background: var(--bg) !important; }
      /* stChatInput contient plusieurs div BaseWeb imbriqués (sans data-testid, fond
         clair #f0f2f6 par défaut) entre le conteneur et le <textarea> : le texte clair
         se retrouvait sur fond clair, quasi invisible. On neutralise tous les fonds
         internes puis on ré-affirme (après, donc prioritaire à spécificité égale) le
         fond de la barre et du bouton d'envoi. */
      /* Idem pour la bordure : au focus, Streamlit colore un de ces div internes en
         rouge (#ff4b4b, l'accent par défaut) — invisible pour le conteneur lui-même,
         donc un simple `border` sur stChatInput ne suffisait pas à l'écraser. */
      [data-testid="stChatInput"] * { background-color: transparent !important; border-color: transparent !important; }
      [data-testid="stChatInput"] {
        background: #ffffff !important; border: 1px solid var(--line) !important;
        border-radius: 22px !important; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
      }
      [data-testid="stChatInput"]:focus-within {
        border-color: var(--text) !important; box-shadow: 0 2px 14px rgba(0,0,0,0.1);
      }
      [data-testid="stChatInputTextArea"] {
        background: transparent !important; color: var(--text) !important; border: none !important;
        font-size: 0.92rem !important;
      }
      [data-testid="stChatInputTextArea"]::placeholder { color: var(--muted) !important; }
      /* stChatInputSubmitButton EST le <button> (pas un div qui le contient) : le
         sélecteur "...SubmitButton button" ne matchait donc rien, et le fond
         restait transparent (écrasé par le reset générique ci-dessus) — icône
         blanche invisible sur fond blanc. */
      [data-testid="stChatInputSubmitButton"] {
        border-radius: 50% !important; background: var(--accent) !important;
        opacity: 1 !important; transition: background-color 0.15s ease;
      }
      [data-testid="stChatInputSubmitButton"]:disabled { background: var(--line) !important; }
      [data-testid="stChatInputSubmitButton"] svg { color: var(--accent-contrast) !important; fill: currentColor !important; }

      /* ── Blocs de code : gris clair, coins arrondis, bouton copier discret ── */
      [data-testid="stCode"] { border-radius: 10px; border: 1px solid var(--line); overflow: hidden; }
      [data-testid="stCode"] pre { background: var(--panel) !important; padding: 1rem !important; }
      [data-testid="stElementToolbar"] button { opacity: 0.55; transition: opacity 0.15s ease; }
      [data-testid="stElementToolbar"] button:hover { opacity: 1; }
      code { background: var(--hover); border-radius: 6px; padding: 0.15rem 0.4rem; color: var(--text); }

      /* ── Indicateur de frappe (3 points) pendant le traitement CPS ───────── */
      [data-testid="stSpinner"] { display: inline-flex; align-items: center; color: var(--muted); font-size: 0.9rem; }
      [data-testid="stSpinner"]::after {
        content: ""; display: inline-block; width: 4px; height: 4px; margin-left: 10px;
        border-radius: 50%; background: var(--muted);
        box-shadow: 9px 0 var(--muted), 18px 0 var(--muted);
        animation: cpb-dots 1.1s infinite ease-in-out;
      }
      @keyframes cpb-dots { 0%, 80%, 100% { opacity: 0.25; } 40% { opacity: 1; } }

      /* ── Volet "Détails de confidentialité" + métriques ──────────────────── */
      [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
      [data-testid="stExpander"] summary:hover { background: var(--hover); }
      [data-testid="stMetricValue"] { color: var(--text); }
      [data-testid="stMetricLabel"] { color: var(--muted); }

      /* ── Responsive : la sidebar devient un drawer/overlay sous 640px ────── */
      @media (max-width: 640px) {
        [data-testid="stSidebar"][aria-expanded="true"] {
          position: fixed; z-index: 999; height: 100vh;
          width: 85vw !important; min-width: 85vw !important;
          box-shadow: 4px 0 24px rgba(0,0,0,0.12);
        }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
          max-width: 88%;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Chargement du système CPS (mise en cache une fois par collection)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Chargement du modèle d'embedding…")
def get_embedder():
    from embeddings.embedder import Embedder
    return Embedder()


@st.cache_resource(show_spinner="Chargement du système CPS (une seule fois)…")
def load_cpb(collection: str, persist_dir: str):
    from rag.hybrid_rag import HybridRAG
    from llms.llama_llm import LlamaLLM
    from countermeasure_v6.cpb_naive_rag_v6 import CPBNaiveRAGV6

    store = IsolatedChromaStore(
        collection_name=collection, persist_dir=persist_dir, embedder=get_embedder()
    )
    if store.count() == 0:
        raise ValueError(
            f"La collection « {collection} » est vide. Lance d'abord l'interface 1 "
            "(ingestion) pour découper et indexer un dataset dans cette collection."
        )
    # dedup=False → plusieurs chunks par doc (config nodedup, comme app.py).
    hybrid = HybridRAG(store=store, llm=LlamaLLM(), dedup=False)
    return CPBNaiveRAGV6(naive_rag=hybrid)


# ══════════════════════════════════════════════════════════════════════════════
# État des sessions (barre latérale façon ChatGPT)
# ══════════════════════════════════════════════════════════════════════════════
def _new_session() -> str:
    sid = str(uuid4())
    st.session_state.sessions[sid] = {
        "title": "Nouvelle conversation",
        "messages": [],
        "cpb_session_id": str(uuid4()),
    }
    st.session_state.active = sid
    return sid


if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "active" not in st.session_state or st.session_state.active not in st.session_state.sessions:
    st.session_state.active = None
if not st.session_state.sessions:
    _new_session()


with st.sidebar:
    st.markdown("### Conversations")
    if st.button("Nouvelle conversation", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    for sid, sess in reversed(list(st.session_state.sessions.items())):
        is_active = sid == st.session_state.active
        label = sess["title"][:34] + ("…" if len(sess["title"]) > 34 else "")
        if st.button(
            ("• " if is_active else "") + label,
            key=f"sess_{sid}",
            use_container_width=True,
        ):
            st.session_state.active = sid
            st.rerun()

    st.divider()
    st.markdown("### Base & réglages")
    collection = st.text_input("Collection", value=DEFAULT_COLLECTION)
    persist_dir = st.text_input("Dossier de la base", value=str(DEFAULT_PERSIST_DIR))
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=TOP_K)


# ══════════════════════════════════════════════════════════════════════════════
# Chargement CPS + entête
# ══════════════════════════════════════════════════════════════════════════════
st.title("Assistant RAG protégé — CPS")
st.markdown(
    '<p class="app-sub">Retrieval hybride (dense + BM25) · requête et chunks bruts '
    "(jamais masqués) · masquage sélectif de la réponse après génération.</p>",
    unsafe_allow_html=True,
)

try:
    cpb_rag = load_cpb(collection.strip() or DEFAULT_COLLECTION,
                       persist_dir.strip() or str(DEFAULT_PERSIST_DIR))
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Bandeau d'info : domaine détecté + combinaisons ré-identifiantes actives.
domain = getattr(cpb_rag.bootstrap_result, "domain", "?")
risky_combos = getattr(cpb_rag, "risky_combos", [])
with st.expander(f"Contexte CPS — domaine détecté : {domain}", expanded=False):
    if risky_combos:
        st.caption("Combinaisons de types masquées dans la RÉPONSE dès qu'elles y sont toutes présentes :")
        for i, combo in enumerate((sorted(c) for c in risky_combos), 1):
            if len(combo) == 1:
                st.markdown(f"**{i}.** `{combo[0]}` (identifiant fort, masqué seul)")
            else:
                st.markdown(f"**{i}.** " + " + ".join(f"`{t}`" for t in combo))
    else:
        st.caption("Aucune combinaison générée → repli sur le masquage v6 standard (poids + hints de domaine).")


# ══════════════════════════════════════════════════════════════════════════════
# Rendu de la conversation active
# ══════════════════════════════════════════════════════════════════════════════
session = st.session_state.sessions[st.session_state.active]


def _render_privacy_details(meta: dict):
    with st.expander("Détails de confidentialité"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks récupérés", meta.get("n_retrieved", 0))
        c2.metric("Remplacements PII (réponse)", meta.get("n_replacements", 0))
        c3.metric("Décision", meta.get("decision", "?"))
        if meta.get("sad_detected"):
            st.warning(f"SAD détecté (B6) — décision : {meta.get('sad_decision', '?')}")


for msg in session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_privacy_details(msg["meta"])


# ══════════════════════════════════════════════════════════════════════════════
# Saisie utilisateur → pipeline CPS
# ══════════════════════════════════════════════════════════════════════════════
def _is_blocked(result: dict) -> bool:
    audit = result.get("cpb_audit")
    if getattr(audit, "decision", "") == "direct_suppression":
        return True
    return result.get("cpb_sad_detected") and result.get("cpb_sad_decision") == "block"


prompt = st.chat_input("Pose ta question…")
if prompt:
    session["messages"].append({"role": "user", "content": prompt})
    if session["title"] == "Nouvelle conversation":
        session["title"] = prompt.strip()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Traitement à travers la contre-mesure CPS…"):
            # session_id propre à la conversation → scoring de risque multi-tour.
            cpb_rag.session_id = session["cpb_session_id"]
            result = cpb_rag.run(prompt.strip(), top_k=top_k)

        response = result.get("response", "")
        if _is_blocked(result):
            st.error(response)
        else:
            st.markdown(response)

        audit = result.get("cpb_audit")
        meta = {
            "query": prompt.strip(),
            "n_retrieved": len(result.get("raw_chunks", []) or []),
            "n_replacements": result.get("cpb_response_n_replacements", 0),
            "sad_detected": bool(result.get("cpb_sad_detected")),
            "sad_decision": result.get("cpb_sad_decision", "pass"),
            "decision": getattr(audit, "decision", "?"),
        }
        _render_privacy_details(meta)

    session["messages"].append({"role": "assistant", "content": response, "meta": meta})
    st.rerun()
