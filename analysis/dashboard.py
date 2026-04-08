# analysis/dashboard.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.tracking
from datetime import datetime
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# --- Config page ---
st.set_page_config(
    page_title="RAG Privacy Benchmark",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Chargement des runs MLflow ---
@st.cache_data(ttl=10)  # refresh toutes les 10 secondes
def load_runs() -> pd.DataFrame:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if experiment is None:
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )

        data = []
        for run in runs:
            data.append({
                "run_id":            run.info.run_id,
                "llm":               run.data.params.get("llm", ""),
                "rag":               run.data.params.get("rag_architecture", ""),
                "attack":            run.data.params.get("attack", ""),
                "tokens_prompt":     run.data.metrics.get("tokens_prompt", 0),
                "tokens_completion": run.data.metrics.get("tokens_completion", 0),
                "tokens_total":      run.data.metrics.get("tokens_total", 0),
                "cost_usd":          run.data.metrics.get("cost_usd", 0.0),
                "pii_leakage_rate":  run.data.metrics.get("pii_leakage_rate", 0.0),
                "rouge_l":           run.data.metrics.get("rouge_l"),
                "auc_roc":           run.data.metrics.get("auc_roc"),
                "jailbreak_success": run.data.metrics.get("jailbreak_success"),
                "start_time":        datetime.fromtimestamp(
                                         run.info.start_time / 1000
                                     ).strftime("%H:%M:%S"),
            })

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Erreur MLflow : {e}")
        return pd.DataFrame()


# ============================================================
#  SIDEBAR
# ============================================================
st.sidebar.title("🔐 RAG Privacy Benchmark")
st.sidebar.markdown("---")

# Bouton refresh manuel
if st.sidebar.button("🔄 Rafraîchir"):
    st.cache_data.clear()

# Filtres
st.sidebar.markdown("### Filtres")
df_all = load_runs()

if df_all.empty:
    st.warning("⚠️ Aucun run MLflow trouvé. Lance d'abord `test_naive_rag.py`.")
    st.stop()

llm_options = ["Tous"] + sorted(df_all["llm"].unique().tolist())
rag_options = ["Tous"] + sorted(df_all["rag"].unique().tolist())
attack_options = ["Tous"] + sorted(df_all["attack"].unique().tolist())

selected_llm    = st.sidebar.selectbox("LLM",          llm_options)
selected_rag    = st.sidebar.selectbox("Architecture",  rag_options)
selected_attack = st.sidebar.selectbox("Attaque",       attack_options)

# Appliquer les filtres
df = df_all.copy()
if selected_llm    != "Tous": df = df[df["llm"]    == selected_llm]
if selected_rag    != "Tous": df = df[df["rag"]     == selected_rag]
if selected_attack != "Tous": df = df[df["attack"]  == selected_attack]

# ============================================================
#  HEADER
# ============================================================
st.title("🔐 RAG Privacy Benchmark — Dashboard temps réel")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')} — "
           f"Refresh automatique toutes les 10s")

# ============================================================
#  KPIs
# ============================================================
st.markdown("### Vue d'ensemble")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Runs total",        len(df_all))
col2.metric("Runs filtrés",      len(df))
col3.metric("Tokens cumulés",    f"{int(df['tokens_total'].sum()):,}")
col4.metric("Coût total (USD)",  f"${df['cost_usd'].sum():.4f}")
col5.metric("Fuite PII moyenne", f"{df['pii_leakage_rate'].mean():.3f}")

st.markdown("---")

# ============================================================
#  HEATMAP VULNÉRABILITÉ
# ============================================================
st.markdown("### 🗺️ Heatmap — Taux de fuite PII (LLM × Architecture)")

df_heat = df_all[df_all["attack"] == "baseline"] if "baseline" in df_all["attack"].values else df_all

if len(df_heat) >= 2:
    try:
        matrix = df_heat.pivot_table(
            index="rag",
            columns="llm",
            values="pii_leakage_rate",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",   # rouge = dangereux, vert = safe
            vmin=0, vmax=1,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Taux de fuite PII moyen par combinaison LLM × RAG")
        ax.set_xlabel("LLM")
        ax.set_ylabel("Architecture RAG")
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.info(f"Heatmap disponible après plus de runs : {e}")
else:
    st.info("⏳ Heatmap disponible après au moins 2 combinaisons LLM × RAG.")

st.markdown("---")

# ============================================================
#  TOKENS ET COÛT
# ============================================================
st.markdown("### 💰 Tokens & Coût par LLM")

col1, col2 = st.columns(2)

with col1:
    tokens_by_llm = df_all.groupby("llm")["tokens_total"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(tokens_by_llm["llm"], tokens_by_llm["tokens_total"], color="#4C72B0")
    ax.set_title("Tokens moyens par run")
    ax.set_ylabel("Tokens")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    st.pyplot(fig)
    plt.close()

with col2:
    cost_by_llm = df_all.groupby("llm")["cost_usd"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(cost_by_llm["llm"], cost_by_llm["cost_usd"], color="#DD8452")
    ax.set_title("Coût cumulé par LLM (USD)")
    ax.set_ylabel("USD")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ============================================================
#  TAUX DE FUITE PAR ATTAQUE
# ============================================================
st.markdown("### ⚔️ Taux de fuite PII par attaque")

if df_all["attack"].nunique() > 1:
    leak_by_attack = df_all.groupby("attack")["pii_leakage_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(
        leak_by_attack["attack"],
        leak_by_attack["pii_leakage_rate"],
        color="#C44E52",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Taux de fuite PII moyen")
    ax.set_title("Vulnérabilité moyenne par type d'attaque")
    for bar, val in zip(bars, leak_by_attack["pii_leakage_rate"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    st.pyplot(fig)
    plt.close()
else:
    st.info("⏳ Disponible après avoir lancé les attaques.")

st.markdown("---")

# ============================================================
#  TABLEAU DES RUNS
# ============================================================
st.markdown("### 📋 Détail des runs")

cols_display = [
    "start_time", "llm", "rag", "attack",
    "tokens_total", "cost_usd",
    "pii_leakage_rate", "rouge_l", "auc_roc",
]
cols_display = [c for c in cols_display if c in df.columns]

st.dataframe(
    df[cols_display].style.background_gradient(
        subset=["pii_leakage_rate"],
        cmap="RdYlGn_r",
        vmin=0, vmax=1,
    ).format({
        "cost_usd":         "${:.6f}",
        "pii_leakage_rate": "{:.4f}",
        "rouge_l":          lambda x: f"{x:.4f}" if x else "—",
        "auc_roc":          lambda x: f"{x:.4f}" if x else "—",
        "tokens_total":     "{:,.0f}",
    }),
    use_container_width=True,
    height=400,
)

# ============================================================
#  AUTO REFRESH
# ============================================================
# Refresh automatique toutes les 10 secondes
import time
time.sleep(10)
st.rerun()