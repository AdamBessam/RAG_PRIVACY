import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from config import TOP_K
from countermeasure.sad_detector import DEFAULT_SBERT_THRESHOLD
from countermeasure_v5.cpb_naive_rag_v5_combo import CPBNaiveRAGV5Combo
from llms.llama_llm import LlamaLLM
from metrics.pii_leakage import compute_pii_leakage
from metrics.response_quality import compute_response_quality
from rag.hybrid_rag import HybridRAG
from vectorstore.chroma_store import ChromaStore

st.set_page_config(page_title="CPB Privacy Shield", layout="wide", page_icon="🔒")

st.title("CPB Privacy Shield")
st.caption("RAG system with Contextual Privacy Budget countermeasure")


@st.cache_resource(show_spinner="Loading CPB v5 system (first time only)...")
def load_system():
    store = ChromaStore()
    llm = LlamaLLM()
    # Retrieval hybride dense (ChromaDB cosinus) + BM25, fusion RRF — comme le
    # runner run_metrics_by_query_type_cedh.py. dedup=False → config nodedup
    # (plusieurs chunks par doc).
    hybrid = HybridRAG(store=store, llm=llm, dedup=False)
    cpb_rag = CPBNaiveRAGV5Combo(naive_rag=hybrid)
    return cpb_rag


def sanitize_chunks(chunks):
    """Les hits BM25 de HybridRAG ont similarity_score=None → max() plante dans
    compute_response_quality / compute_pii_leakage. On remplace None par le
    rrf_score (ou 0.0) sur une COPIE, sans muter les chunks d'origine."""
    safe = []
    for c in chunks or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        if cc.get("similarity_score") is None:
            cc["similarity_score"] = float(cc.get("rrf_score") or 0.0)
        safe.append(cc)
    return safe


cpb_rag = load_system()

# ── BLOC B0 — Domaine + combinaisons risquées (LLM, calculé une fois) ───────────
domain = getattr(cpb_rag.bootstrap_result, "domain", "?")
risky_combos = getattr(cpb_rag, "risky_combos", [])

st.subheader("BLOC B0 · Combinaisons de données sensibles ré-identifiantes")
st.markdown(
    f"**Domaine détecté :** `{domain}` &nbsp;—&nbsp; "
    f"**{len(risky_combos)} combinaison(s)** générée(s) par le LLM pour ce corpus."
)
if risky_combos:
    st.caption(
        "Une entité seule est souvent inoffensive ; c'est l'ASSEMBLAGE de ces types "
        "dans un même passage qui ré-identifie une personne. CPB v5 masque tous les "
        "membres d'une combinaison dès qu'ils sont tous présents dans un chunk."
    )
    for i, combo in enumerate(risky_combos, 1):
        members = sorted(combo)
        if len(members) == 1:
            st.markdown(f"**{i}.** `{members[0]}` &nbsp;(identifiant fort — masqué seul)")
        else:
            st.markdown(f"**{i}.** " + " + ".join(f"`{t}`" for t in members))
else:
    st.warning(
        "Aucune combinaison générée (LLM off ou échec) → repli sur le masquage v5 standard."
    )

st.divider()

# ── Query input ────────────────────────────────────────────────────────────────
with st.form("query_form"):
    query = st.text_area("Query", placeholder="Enter your question here...", height=80)
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=TOP_K)
    submitted = st.form_submit_button("Submit", use_container_width=True)

if not submitted or not query.strip():
    st.stop()

# ── Run CPB ────────────────────────────────────────────────────────────────────
with st.spinner("Processing query through CPB pipeline..."):
    result = cpb_rag.run(query.strip(), top_k=top_k)

# HybridRAG : neutralise similarity_score=None (hits BM25) avant les métriques.
metric_chunks = sanitize_chunks(result["raw_chunks"])

pii = compute_pii_leakage(response=result["response"], chunks=metric_chunks, query=query.strip())

with st.spinner("Computing response quality metrics..."):
    quality = compute_response_quality(
        query=query.strip(),
        response=result["response"],
        chunks=metric_chunks,
        embedder=cpb_rag.store.embedder,
    )

# ── Response ───────────────────────────────────────────────────────────────────
decision = result.get("cpb_response_guard_decision", "")
sad_detected = result.get("cpb_sad_detected", False)

if decision in ("direct_suppression", "all_chunks_suppressed", "exception") or (
    sad_detected and result.get("cpb_sad_decision") == "block"
):
    st.error(result["response"])
else:
    st.success(result["response"])

st.divider()

# ── Summary metrics ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="PII Leakage Rate",
        value=f"{pii.leakage_rate:.0%}",
        delta=f"{pii.n_pii_leaked} / {pii.n_pii_total} entities",
        delta_color="inverse",
    )

with col2:
    query_risk = result.get("cpb_query_risk", 0.0)
    st.metric(
        label="Query Risk",
        value=f"{query_risk:.3f}",
        delta="suppressed" if decision == "direct_suppression" else "passed",
        delta_color="inverse" if decision == "direct_suppression" else "normal",
    )

with col3:
    sad_decision = result.get("cpb_sad_decision", "pass")
    sad_conf = result.get("cpb_sad_confidence", 0.0)
    sad_label = "DETECTED" if sad_detected else "Clean"
    st.metric(
        label="SAD Detection",
        value=sad_label,
        delta=f"{sad_decision}  conf={sad_conf:.2f}",
        delta_color="inverse" if sad_detected else "normal",
    )

with col4:
    chunk_decisions = result.get("cpb_chunk_decisions", [])
    n_masked = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "mask")
    n_suppressed = sum(1 for d in chunk_decisions if getattr(d, "decision", "") == "suppress")
    st.metric(
        label="Chunks",
        value=f"{n_masked} masked",
        delta=f"{n_suppressed} suppressed",
        delta_color="inverse" if n_suppressed > 0 else "normal",
    )

# ── Quality metrics ────────────────────────────────────────────────────────────
st.subheader("Response Quality")
qcol1, qcol2, qcol3, qcol4 = st.columns(4)

with qcol1:
    score = quality.quality_score
    st.metric("Quality Score", f"{score:.3f}", delta="good" if score >= 0.5 else "low",
              delta_color="normal" if score >= 0.5 else "inverse")

with qcol2:
    st.metric("Answer Relevancy", f"{quality.answer_relevancy:.3f}",
              help="Cosine similarity between query and response embeddings")

with qcol3:
    st.metric("ROUGE-L", f"{quality.rouge_l:.3f}",
              help="Textual overlap between response and best retrieved chunk")

with qcol4:
    if quality.bert_score_f1 > 0:
        st.metric("BERTScore F1", f"{quality.bert_score_f1:.3f}",
                  help="Semantic similarity between response and best retrieved chunk")
    else:
        st.metric("BERTScore F1", "N/A", help="bert_score not available")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# CPB PIPELINE TRACE
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("CPB Pipeline Trace")

tab1a, tab2, tab3, tab4, tab6, tab5b = st.tabs([
    "1A · Query Risk",
    "2 · Query PII",
    "3 · Budget Gate",
    "4 · Chunks",
    "6 · SAD Detector",
    "5b · Response Guard",
])

# ── BLOC 1A — Query Risk Scorer ───────────────────────────────────────────────
with tab1a:
    signals = result.get("cpb_query_risk_signals", {})
    ner_entities = result.get("cpb_ner_entities", [])

    st.markdown(f"**Score total r = `{query_risk:.4f}`** &nbsp; (seuil suppression directe = 0.85)")

    SIGNAL_META = {
        "s1_ner":        ("S1  NER (spaCy)",             0.15),
        "s2_extractive": ("S2  Extractif / contexte",    0.25),
        "s3_jailbreak":  ("S3  Jailbreak",               0.35),
        "s4_session":    ("S4  Session multi-tour",      0.10),
        "s5_semantic":   ("S5  Sémantique SBERT",        0.15),
    }

    st.markdown("**Signaux de risque**")
    for key, (label, max_w) in SIGNAL_META.items():
        val = signals.get(key, 0.0)
        pct = val / max_w if max_w else 0.0
        cols = st.columns([3, 1, 1])
        cols[0].progress(min(pct, 1.0), text=label)
        cols[1].markdown(f"`{val:.4f}`")
        cols[2].markdown(f"/ {max_w:.2f}")

    st.markdown("**Entités NER détectées**")
    if ner_entities:
        for e in ner_entities:
            st.markdown(f"- `[{e['label']}]` **{e['text']}** &nbsp; pos {e['start']}–{e['end']}")
    else:
        st.info("Aucune entité NER détectée dans la query.")

# ── BLOC 2 — Query PII ────────────────────────────────────────────────────────
with tab2:
    query_pii_score = result.get("cpb_query_pii_score", 0.0)
    query_pii_findings = result.get("cpb_query_pii_findings", [])
    masked_query = result.get("cpb_masked_query", query.strip())

    st.markdown(f"**Score PII query p = `{query_pii_score:.4f}`**")

    if query_pii_findings:
        st.markdown(f"**{len(query_pii_findings)} PII trouvé(s) dans la query :**")
        for f in query_pii_findings:
            st.markdown(f"- `[{f.entity_type}]` **{f.text}** &nbsp; score Presidio = `{f.score:.2f}`")
        st.markdown("**Query masquée envoyée au RAG :**")
        st.code(masked_query, language=None)
    else:
        st.success("Aucun PII détecté dans la query — envoyée telle quelle.")
        st.code(query.strip(), language=None)

    replacements = result.get("cpb_query_pii_replacements", 0)
    if replacements:
        st.caption(f"{replacements} remplacement(s) effectué(s)")

# ── BLOC 3 — Budget Gate (suppression directe) ────────────────────────────────
with tab3:
    is_direct = decision == "direct_suppression"
    s3 = signals.get("s3_jailbreak", 0.0)

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Suppression directe (r > 0.85)",
            "OUI" if query_risk > 0.85 else "NON",
            delta=f"r = {query_risk:.4f}",
            delta_color="inverse" if query_risk > 0.85 else "normal",
        )
    with c2:
        st.metric(
            "Jailbreak immédiat (s3 > 0)",
            "OUI" if s3 > 0 else "NON",
            delta=f"s3 = {s3:.4f}",
            delta_color="inverse" if s3 > 0 else "normal",
        )

    if is_direct:
        st.error("Query bloquée ici — pipeline arrêté.")
    else:
        st.success("Query non bloquée — passage au retrieval.")

    st.markdown("**Formule budget par chunk :** `b = 1 – (r × p)`")
    st.markdown(f"- Seuil suppression chunk : `b < 0.40`")
    st.markdown(f"- r actuel : `{query_risk:.4f}`")

# ── BLOC 4 — Chunks ───────────────────────────────────────────────────────────
with tab4:
    raw_chunks = result.get("raw_chunks", [])
    safe_chunks = result.get("chunks", [])

    st.markdown(
        f"**{len(raw_chunks)} chunks récupérés → {len(safe_chunks)} envoyés au LLM**"
    )

    for i, (d, raw) in enumerate(zip(chunk_decisions, raw_chunks), 1):
        dec = getattr(d, "decision", "?")
        p = getattr(d, "pii_score", 0.0)
        b = getattr(d, "budget", 0.0)
        n_f = getattr(d, "n_findings", 0)
        color = "🔴" if dec == "suppress" else "🟡"

        with st.expander(f"{color} Chunk {i} — `{dec.upper()}` | b = {b:.3f} | {n_f} PII"):
            st.markdown(
                f"**Budget :** `b = 1 – ({query_risk:.3f} × {p:.3f}) = {b:.3f}`"
                f"  →  seuil = 0.40"
            )
            st.markdown(f"**PII score p :** `{p:.4f}`")

            # Texte brut
            raw_text = raw.get("text", "")
            st.markdown("**Texte brut (avant masquage) :**")
            st.code(raw_text[:500] + ("..." if len(raw_text) > 500 else ""), language=None)

            # Texte masqué (si dans safe_chunks)
            safe_match = next(
                (c for c in safe_chunks if c.get("chunk_id") == raw.get("chunk_id")), None
            )
            if safe_match:
                masked_text = safe_match.get("text", "")
                st.markdown("**Texte masqué envoyé au LLM :**")
                st.code(masked_text[:500] + ("..." if len(masked_text) > 500 else ""), language=None)
                findings_info = safe_match.get("cpb_findings", [])
                if findings_info:
                    st.markdown(f"**Entités masquées ({len(findings_info)}) :**")
                    for f in findings_info:
                        st.markdown(
                            f"- `[{f['entity_type']}]` **{f['text']}** "
                            f"→ `[{f['entity_type']}_N]`"
                        )
            elif dec == "suppress":
                st.error("Chunk supprimé — non envoyé au LLM.")

# ── BLOC 6 — SAD Detector ─────────────────────────────────────────────────────
with tab6:
    sad_result = result.get("cpb_sad_result")
    sad_filter = result.get("cpb_sad_filter", 0)
    sad_cats = result.get("cpb_sad_categories", [])
    sad_conf = result.get("cpb_sad_confidence", 0.0)
    max_sim = getattr(sad_result, "max_similarity", 0.0) if sad_result else 0.0
    category_scores = getattr(sad_result, "sbert_category_scores", {}) if sad_result else {}

    # F1
    st.markdown("### F1 — Regex : sujet individuel")
    from countermeasure.sad_detector import INDIVIDUAL_RE
    match_resp = INDIVIDUAL_RE.search(result.get("response", ""))
    match_query = INDIVIDUAL_RE.search(query.strip())
    c1, c2 = st.columns(2)
    with c1:
        if match_resp:
            st.success(f"Trouvé dans la réponse : `{match_resp.group()}`")
        else:
            st.warning("Aucun sujet individuel dans la réponse")
    with c2:
        if match_query:
            st.success(f"Trouvé dans la query : `{match_query.group()}`")
        else:
            st.warning("Aucun sujet individuel dans la query")

    if not match_resp and not match_query:
        st.error("Cascade arrêtée en F1 — réponse considérée safe (pass)")
    else:
        st.info("F1 passe → F2 exécuté")

    # F2
    st.markdown(f"### F2 — SBERT centroïde (seuil = {DEFAULT_SBERT_THRESHOLD})")
    if category_scores:
        for cat, sim in sorted(category_scores.items(), key=lambda x: -x[1]):
            above = sim >= DEFAULT_SBERT_THRESHOLD
            bar_val = min(sim / DEFAULT_SBERT_THRESHOLD, 1.0) if DEFAULT_SBERT_THRESHOLD else 0.0
            cols = st.columns([2, 4, 1])
            label_text = f"**{cat}**" + (" ← ABOVE THRESHOLD" if above else "")
            cols[0].markdown(label_text)
            cols[1].progress(min(bar_val, 1.0))
            cols[2].markdown(f"`{sim:.4f}`")
    else:
        st.warning("Scores SBERT non disponibles (SAD non exécuté ou F1 bloqué).")

    st.caption(f"Max similarité globale : `{max_sim:.4f}` | Seuil : `{DEFAULT_SBERT_THRESHOLD}`")

    # F3
    st.markdown("### F3 — Phi-3 Mini (juge LLM)")
    if sad_result:
        fcols = st.columns(3)
        fcols[0].metric("SAD détecté", "OUI" if sad_detected else "NON")
        fcols[1].metric("Confiance", f"{sad_conf:.2f}", help="Seuil mask = 0.70")
        fcols[2].metric("Filtre déclenché", f"F{sad_filter}")

        if sad_detected:
            st.warning(f"Catégorie(s) : **{', '.join(sad_cats)}** — décision : `{sad_decision}`")
        else:
            st.success("Phi-3 n'a pas détecté de SAD.")

        if sad_result.reasoning:
            st.markdown(f"**Raisonnement Phi-3 :** {sad_result.reasoning}")
    else:
        st.info("SAD non exécuté (query bloquée avant).")

    # Réponse après SAD
    st.markdown("### Réponse après SAD")
    sad_response = getattr(sad_result, "response", result.get("response", "")) if sad_result else result.get("response", "")
    if sad_detected and sad_decision == "block":
        st.error(sad_response)
    elif sad_detected:
        st.warning(sad_response)
    else:
        st.info(sad_response)

# ── BLOC 5b — Response Guard ──────────────────────────────────────────────────
with tab5b:
    guard = result.get("cpb_response_guard")
    guard_decision = result.get("cpb_response_guard_decision", "")

    DECISION_COLORS = {
        "reliable":    st.success,
        "fix":         st.warning,
        "reask":       st.warning,
        "reask_fix":   st.warning,
        "exception":   st.error,
        "direct_suppression": st.error,
        "all_chunks_suppressed": st.error,
    }
    render = DECISION_COLORS.get(guard_decision, st.info)

    if guard:
        c1, c2, c3 = st.columns(3)
        c1.metric("Décision guard", guard_decision)
        c2.metric("PII findings", guard.n_findings)
        c3.metric("Leakage score", f"{guard.leakage_score:.4f}")

        if guard.reason:
            st.caption(f"Raison : {guard.reason}")

        if guard.n_findings == 0:
            st.info(
                "Presidio n'a détecté aucun PII dans la réponse. "
                "Note : Presidio reconnaît PERSON, LOCATION, EMAIL, PHONE, etc. "
                "mais pas les noms de substances (ex. heroin) ni les attributs sensibles."
            )
        else:
            st.markdown(f"**{guard.n_findings} PII détecté(s)** — {guard.n_replacements} remplacement(s)")

        render(f"Réponse finale : {result['response']}")
    else:
        st.info("Response Guard non exécuté (query bloquée avant).")

# ── Leaked PII detail ──────────────────────────────────────────────────────────
if pii.leaked_entities:
    with st.expander(f"Leaked PII entities ({pii.n_pii_leaked})"):
        for ent in pii.leaked_entities:
            st.write(f"- **{ent['text']}** — type: `{ent['type']}` | sensitivity: `{ent['sensitivity']}`")
