import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="Test Contre-Mesure CPB",
    layout="wide",
)

# ─── Initialisation session ────────────────────────────────────────────────────
_DEFAULTS = {
    "indexed": False,
    "store": None,
    "cpb_rag": None,
    "history": [],
    "chunk_count": 0,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── En-tête ──────────────────────────────────────────────────────────────────
st.title("Test Contre-Mesure CPB")
st.caption(
    "Chargez n'importe quel dataset, posez une question et observez la réponse "
    "avec la protection CPB (Contextual Privacy Budget)."
)
st.divider()


# ─── Sidebar : ingestion ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dataset")
    st.markdown("Formats supportés : **PDF · TXT · CSV · JSON · XLSX · XLS**")

    method = st.radio("Source", ["Upload fichier", "Chemin local"], horizontal=True)

    uploaded = None
    local_path = ""

    if method == "Upload fichier":
        uploaded = st.file_uploader(
            "Glissez-deposez votre fichier",
            type=["pdf", "txt", "csv", "json", "xlsx", "xls"],
        )
    else:
        local_path = st.text_input(
            "Chemin vers le fichier",
            placeholder="C:/Users/.../mon_dataset.txt",
        )

    if st.button("Charger et indexer", type="primary", use_container_width=True):
        from store import TestChromaStore
        from ingest import load_text_from_path, load_text_from_upload, chunk_text
        from rag.naive_rag import NaiveRAG
        from countermeasure.cpb_naive_rag import CPBNaiveRAG
        from llms.llama_llm import LlamaLLM

        err = None
        if method == "Upload fichier" and uploaded is None:
            err = "Veuillez uploader un fichier."
        elif method == "Chemin local" and not local_path.strip():
            err = "Veuillez entrer un chemin valide."

        if err:
            st.error(err)
        else:
            with st.status("Traitement en cours...", expanded=True) as status:
                try:
                    st.write("Lecture du fichier...")
                    if method == "Upload fichier":
                        text = load_text_from_upload(uploaded.read(), uploaded.name)
                    else:
                        text = load_text_from_path(local_path.strip())
                    st.write(f"{len(text):,} caracteres charges")

                    st.write("Decoupe en chunks...")
                    chunks = chunk_text(text)
                    st.write(f"{len(chunks)} chunks crees")

                    st.write("Generation des embeddings et indexation...")
                    store = TestChromaStore()
                    store.reset()
                    store.index_chunks(chunks)
                    st.write("Indexe dans ChromaDB")

                    st.write("Initialisation du pipeline RAG avec CPB...")
                    llm = LlamaLLM()
                    naive = NaiveRAG(store=store, llm=llm)
                    cpb = CPBNaiveRAG(naive_rag=naive)

                    st.session_state.store = store
                    st.session_state.cpb_rag = cpb
                    st.session_state.indexed = True
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.history = []

                    status.update(label="Dataset pret !", state="complete")

                except Exception as e:
                    status.update(label="Erreur lors du chargement", state="error")
                    st.error(str(e))

    if st.session_state.indexed:
        st.divider()
        st.metric("Chunks indexes", st.session_state.chunk_count)
        if st.button("Reinitialiser", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()


# ─── Zone principale : chat ───────────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("Chargez un dataset dans la barre laterale pour commencer.")
    st.stop()

st.subheader("Posez une question")

with st.form("qform", clear_on_submit=True):
    question = st.text_area(
        "Votre question",
        placeholder="Que contient ce document ? Qui est mentionne ?",
        height=90,
    )
    top_k = st.slider("Nombre de chunks recuperes (top-k)", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Envoyer", use_container_width=True)

if submitted and question.strip():
    st.markdown("### Reponse avec protection CPB")
    with st.spinner("Generation avec protection CPB..."):
        r_cpb = st.session_state.cpb_rag.run(question, top_k=top_k)

    if r_cpb.get("cpb_sad_detected"):
        st.warning(
            f"SAD detecte · categories : {r_cpb.get('cpb_sad_categories')} "
            f"· decision : {r_cpb.get('cpb_sad_decision')}"
        )

    st.markdown(r_cpb["response"])

    # Metriques CPB
    m1, m2, m3 = st.columns(3)
    m1.metric("Risque requete", f"{r_cpb.get('cpb_query_risk', 0):.2f}")
    decisions = r_cpb.get("cpb_chunk_decisions", [])
    suppressed = sum(1 for d in decisions if hasattr(d, "decision") and d.decision == "suppress")
    masked_n = sum(1 for d in decisions if hasattr(d, "decision") and d.decision == "mask")
    m2.metric("Chunks supprimes", suppressed)
    m3.metric("Chunks masques", masked_n)

    with st.expander(f"Chunks apres filtrage CPB ({len(r_cpb.get('chunks', []))})"):
        for i, c in enumerate(r_cpb.get("chunks", [])):
            st.markdown(f"**Chunk {i+1}** — similarite : `{c['similarity_score']:.3f}`")
            st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))

    st.session_state.history.append({
        "q": question,
        "avec": r_cpb["response"],
        "risk": r_cpb.get("cpb_query_risk", 0),
    })


# ─── Historique ───────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    with st.expander(f"Historique — {len(st.session_state.history)} question(s)"):
        for i, item in enumerate(reversed(st.session_state.history)):
            n = len(st.session_state.history) - i
            st.markdown(f"**Q{n} · Risque CPB : `{item['risk']:.2f}`**")
            st.markdown(f"> {item['q']}")
            st.markdown("**Reponse CPB :**\n\n" + item["avec"][:250] + "…")
            st.divider()
