"""
Interface 1 (Streamlit, ISOLÉE) — Ingestion d'un dataset + analyse B0.

Lancer :   streamlit run interface/app_ingestion_b0.py

Flux :
  1. Choisir une source : fichier CSV, lien Hugging Face, ou texte brut.
  2. Cliquer sur « Chunker, indexer et analyser (B0) ».
  3. Le dataset est découpé en chunks et indexé dans une base vectorielle
     ChromaDB ISOLÉE (dossier + collection dédiés).
  4. L'étape B0 de la countermeasure v6 est exécutée : domaine détecté,
     catégories sensibles, taxonomie, et COMBINAISONS de types ré-identifiantes.

Cette interface ne modifie aucun fichier existant et n'écrit jamais dans la
collection de benchmark. Habillage volontairement épuré, façon ChatGPT.
"""

import sys
from pathlib import Path

# ── Préchargement torch (DOIT rester en tout premier) ─────────────────────────
# Sur Windows, pyarrow (chargé par Streamlit dès le premier st.dataframe) initialise
# des DLL natives qui, si elles sont chargées AVANT torch, cassent l'init de
# c10.dll → OSError [WinError 1114] au chargement de l'embedder. Importer torch
# ici, avant streamlit/pandas/pyarrow, initialise c10.dll en premier et évite le
# crash. IMPORTANT : ce correctif exige un redémarrage COMPLET du process
# `streamlit run` (pas seulement un refresh du navigateur) — sinon pyarrow reste
# chargé en mémoire depuis un rerun précédent et torch échoue quand même.
# On ne tolère QUE l'absence de torch (ImportError) ; une erreur DLL doit remonter
# immédiatement et clairement, plutôt que ressurgir plus tard à l'embedder.
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from interface.ingestion import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    IsolatedChromaStore,
    chunk_documents,
    load_docs_from_csv,
    load_docs_from_huggingface,
    load_docs_from_text,
    run_b0_analysis,
    save_analysis,
)

st.set_page_config(page_title="Ingestion et analyse B0", layout="centered")


# ══════════════════════════════════════════════════════════════════════════════
# Habillage épuré (façon ChatGPT) — colonne centrée, palette neutre, coins arrondis
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
      :root {
        --ink: #0d0d0d;
        --muted: #6e6e80;
        --line: #e5e5e5;
        --panel: #ffffff;
        --bg: #ffffff;
        --accent: #0d0d0d;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --ink: #ececf1;
          --muted: #9a9aa7;
          --line: #2a2a2e;
          --panel: #1e1e20;
          --bg: #131314;
          --accent: #ececf1;
        }
      }

      .stApp { background: var(--bg); }
      .block-container {
        max-width: 46rem;
        padding-top: 3.2rem;
        padding-bottom: 5rem;
      }

      h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }
      h1 { font-size: 1.85rem; font-weight: 650; }
      h2, .stSubheader { font-size: 1.1rem !important; font-weight: 600; }
      p, label, span, li { color: var(--ink); }
      .app-sub { color: var(--muted); font-size: 0.95rem; line-height: 1.5; margin-top: -0.3rem; }

      /* Champs de saisie : coins arrondis, bordure discrète */
      .stTextInput input, .stTextArea textarea, .stNumberInput input {
        border-radius: 12px !important;
        border: 1px solid var(--line) !important;
        background: var(--panel) !important;
        color: var(--ink) !important;
      }
      .stTextArea textarea { min-height: 8rem; }

      /* Bouton principal : sombre, pleine largeur, arrondi (style composer) */
      .stButton > button {
        border-radius: 999px !important;
        border: 1px solid var(--accent) !important;
        background: var(--accent) !important;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: opacity 0.15s ease;
      }
      /* Forcer la couleur du libellé (le <p>/<span> interne héritait du noir global) */
      .stButton > button,
      .stButton > button *,
      .stButton > button p,
      .stButton > button span,
      .stButton > button div { color: var(--bg) !important; }
      .stButton > button:hover { opacity: 0.85; }

      /* Uploader et expanders : cartes douces */
      [data-testid="stFileUploader"] section,
      .streamlit-expanderHeader,
      details {
        border-radius: 12px !important;
        border: 1px solid var(--line) !important;
        background: var(--panel) !important;
      }

      /* Cartes de métriques discrètes */
      [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.9rem 1rem;
      }

      /* Radio en ligne, aéré */
      [role="radiogroup"] { gap: 1.2rem; }

      hr { border-color: var(--line); }
      code { background: rgba(127,127,127,0.12); border-radius: 6px; padding: 0.1rem 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Ingestion et analyse du dataset")
st.markdown(
    '<p class="app-sub">Charge une source de données, découpe-la et indexe-la dans '
    "une base vectorielle isolée, puis lance l'étape B0 de la countermeasure v6 "
    "(domaine, catégories, combinaisons ré-identifiantes).</p>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Chargement du modèle d'embedding…")
def get_embedder():
    from embeddings.embedder import Embedder
    return Embedder()


# ══════════════════════════════════════════════════════════════════════════════
# 1 · Source des données
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Source des données")

source = st.radio(
    "Type de source",
    ["Fichier CSV / XLSX", "Lien Hugging Face", "Texte brut"],
    horizontal=True,
    label_visibility="collapsed",
)

csv_file = None
csv_text_column = None
hf_name = hf_config = hf_split = hf_text_column = None
raw_text = doc_separator = None

if source == "Fichier CSV / XLSX":
    csv_file = st.file_uploader("Fichier CSV ou XLSX du dataset", type=["csv", "xlsx", "xls", "xlsm"])
    if csv_file is not None:
        try:
            from interface.ingestion import read_table
            preview = read_table(csv_file, nrows=50)
            st.dataframe(preview.head(10), use_container_width=True)
            csv_text_column = st.selectbox(
                "Colonne contenant le texte", list(preview.columns)
            )
        except Exception as exc:
            st.error(f"Impossible de lire le fichier : {exc}")

elif source == "Lien Hugging Face":
    hf_name = st.text_input(
        "Dataset Hugging Face (identifiant ou URL)",
        placeholder="ex. ildpil/text-anonymization-benchmark  ou  https://huggingface.co/datasets/ildpil/text-anonymization-benchmark",
        help="Identifiant de dépôt (owner/nom) ou URL complète — les deux sont acceptés.",
    )
    c1, c2, c3 = st.columns(3)
    hf_config = c1.text_input("Config / subset (optionnel)", value="")
    hf_split = c2.text_input("Split", value="train")
    hf_text_column = c3.text_input("Colonne texte", value="text")

else:  # Texte brut
    raw_text = st.text_area(
        "Colle ton texte ici", height=220, placeholder="Un ou plusieurs documents…"
    )
    doc_separator = st.text_input(
        "Séparateur de documents", value="\\n\\n",
        help="Chaque bloc séparé par ce motif devient un document. Défaut : ligne vide.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2 · Paramètres d'indexation (isolée)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Base vectorielle isolée et chunking")

with st.expander("Paramètres avancés", expanded=False):
    col_a, col_b = st.columns(2)
    collection_name = col_a.text_input("Nom de la collection", value=DEFAULT_COLLECTION)
    persist_dir = col_b.text_input("Dossier de la base", value=str(DEFAULT_PERSIST_DIR))
    col_c, col_d, col_e = st.columns(3)
    chunk_size = col_c.number_input("Chunk size", min_value=100, max_value=4000,
                                    value=DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = col_d.number_input("Chunk overlap", min_value=0, max_value=1000,
                                       value=DEFAULT_CHUNK_OVERLAP, step=10)
    max_docs = col_e.number_input("Max documents (0 = tous)", min_value=0, value=0, step=50)
    reset_collection = st.checkbox(
        "Vider la collection avant indexation (repartir de zéro)", value=True
    )

run = st.button("Chunker, indexer et analyser (B0)", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# Exécution
# ══════════════════════════════════════════════════════════════════════════════
def _build_docs():
    limit = int(max_docs) or None
    if source == "Fichier CSV / XLSX":
        if csv_file is None:
            raise ValueError("Charge d'abord un fichier CSV ou XLSX.")
        if not csv_text_column:
            raise ValueError("Choisis la colonne texte du fichier.")
        return load_docs_from_csv(csv_file, csv_text_column, max_docs=limit)
    if source == "Lien Hugging Face":
        if not (hf_name or "").strip():
            raise ValueError("Renseigne le nom du dataset Hugging Face.")
        return load_docs_from_huggingface(
            hf_name.strip(),
            text_column=(hf_text_column or "text").strip(),
            split=(hf_split or "train").strip(),
            config=(hf_config or "").strip() or None,
            max_docs=limit,
        )
    sep = (doc_separator or "\\n\\n").encode().decode("unicode_escape")
    return load_docs_from_text(raw_text or "", separator=sep)


if run:
    try:
        with st.status("Traitement en cours…", expanded=True) as status:
            st.write("Chargement des documents…")
            docs = _build_docs()
            st.write(f"{len(docs)} document(s) chargé(s).")

            st.write("Découpage en chunks…")
            chunks = chunk_documents(docs, chunk_size=int(chunk_size),
                                     chunk_overlap=int(chunk_overlap))
            if not chunks:
                raise ValueError("Aucun chunk produit (documents vides ?).")
            st.write(f"{len(chunks)} chunk(s) créé(s).")

            st.write("Indexation dans la base vectorielle isolée…")
            store = IsolatedChromaStore(
                collection_name=collection_name.strip() or DEFAULT_COLLECTION,
                persist_dir=persist_dir.strip() or str(DEFAULT_PERSIST_DIR),
                embedder=get_embedder(),
            )
            if reset_collection:
                store.reset()
            total = store.index_chunks(chunks)
            st.write(f"{total} chunk(s) dans la collection « {store.collection_name} ».")

            st.write("Analyse B0 (domaine, catégories, combinaisons)…")
            analysis = run_b0_analysis(store)
            json_path = save_analysis(analysis, store.persist_dir)
            st.write(f"Analyse sauvegardée : {json_path}")

            status.update(label="Terminé", state="complete", expanded=False)

        st.session_state["analysis"] = analysis
        st.session_state["ingest_stats"] = {
            "n_docs": len(docs),
            "n_chunks": len(chunks),
            "total_indexed": total,
            "collection": store.collection_name,
            "persist_dir": store.persist_dir,
        }
    except Exception as exc:
        st.error(f"Échec : {exc}")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 3 · Résultats
# ══════════════════════════════════════════════════════════════════════════════
analysis = st.session_state.get("analysis")
stats = st.session_state.get("ingest_stats")

if analysis is None:
    st.info("Configure une source ci-dessus puis lance l'analyse.")
    st.stop()

st.divider()
st.subheader("Ingestion")
if stats:
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", stats["n_docs"])
    m2.metric("Chunks indexés", stats["total_indexed"])
    m3.metric("Collection", stats["collection"])
    st.caption(f"Base vectorielle isolée : `{stats['persist_dir']}`")

# ── Domaine ────────────────────────────────────────────────────────────────────
st.subheader("Analyse B0 · Domaine")
d1, d2, d3 = st.columns(3)
d1.metric("Domaine détecté", analysis.domain or "?")
d2.metric("Confiance", f"{analysis.domain_confidence:.2f}")
d3.metric("Source", analysis.domain_source)
if analysis.used_fallback:
    st.warning("B0 a utilisé un repli (LLM/classifieur indisponible) — résultats dégradés.")

# ── Catégories + taxonomie ──────────────────────────────────────────────────────
st.subheader("Catégories sensibles et taxonomie")
if analysis.categories:
    st.markdown(f"**{len(analysis.categories)} catégorie(s)** générée(s) pour ce domaine :")
    for cat in analysis.categories:
        hints = analysis.category_hints.get(cat, [])
        with st.expander(cat + (f"  ·  types : {', '.join(hints)}" if hints else "")):
            phrases = analysis.taxonomy.get(cat, [])
            if phrases:
                st.markdown("**Phrases d'ancrage (taxonomie) :**")
                for p in phrases:
                    st.markdown(f"- {p}")
            else:
                st.caption("Aucune phrase d'ancrage.")
else:
    st.warning("Aucune catégorie générée (LLM off ou échec).")

# ── Types PII appris ────────────────────────────────────────────────────────────
if analysis.learned_types:
    st.markdown("**Types PII découverts dans le corpus (Presidio) :**")
    st.markdown(" ".join(f"`{t}`" for t in analysis.learned_types))

# ── Combinaisons ré-identifiantes ───────────────────────────────────────────────
st.subheader("Combinaisons de données ré-identifiantes")
st.caption(
    "Une entité seule est souvent inoffensive ; c'est l'assemblage de ces types "
    "dans un même passage qui ré-identifie une personne. CPB v6 masque tous les "
    "membres d'une combinaison dès qu'ils sont tous présents dans un chunk."
)
if analysis.risky_combos:
    for i, combo in enumerate(analysis.risky_combos, 1):
        if len(combo) == 1:
            st.markdown(f"**{i}.** `{combo[0]}` &nbsp;(identifiant fort — masqué seul)")
        else:
            st.markdown(f"**{i}.** " + " + ".join(f"`{t}`" for t in combo))
else:
    st.warning("Aucune combinaison générée (LLM off ou échec) → repli sur le masquage v6 standard.")

with st.expander("Voir le JSON brut de l'analyse B0"):
    st.json(analysis.raw)
