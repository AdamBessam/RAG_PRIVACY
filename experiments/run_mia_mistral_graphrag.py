# experiments/run_mia_mistral_graphrag.py
"""
Lance le RAG-MIA (black-box) sur Mistral × GraphRAG uniquement.

Usage
-----
python experiments/run_mia_mistral_graphrag.py

Résultats loggés dans MLflow : tpr, fpr, auc_roc, n_missing, cost_usd_total
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random

import mlflow

from vectorstore.chroma_store import ChromaStore
from llms.mistral_llm import MistralLLM
from rag.graph_rag import GraphRAG
from attacks.membership_inference import MembershipInferenceAttack, ATTACK_PROMPTS
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, DATASET_NAME
from data.loader import load_raw_documents


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_MEMBERS     = 50
N_NON_MEMBERS = 50
PROMPT_ID     = 2
SEED          = 42
LLM_NAME      = "mistral:7b"
RAG_NAME      = "graph_rag"


def get_member_non_member_texts(store: ChromaStore, all_docs: list[dict]) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    stored     = store.collection.get(include=["metadatas"])
    stored_ids = set(m["doc_id"] for m in stored["metadatas"])

    random.seed(SEED)
    shuffled = list(all_docs)
    random.shuffle(shuffled)

    members = []
    for doc in shuffled:
        if doc["doc_id"] in stored_ids and len(members) < N_MEMBERS:
            members.append(doc["text"].strip())
        if len(members) >= N_MEMBERS:
            break

    val_dataset = load_dataset(DATASET_NAME, split="validation")
    val_docs    = list(val_dataset)
    random.shuffle(val_docs)
    non_members = [d["text"].strip() for d in val_docs[:N_NON_MEMBERS]]

    print(f"✅ Membres trouvés     : {len(members)}")
    print(f"✅ Non-membres trouvés : {len(non_members)}")
    return members, non_members


def run_mia(rag, members: list[str], non_members: list[str]) -> dict:
    print(f"\n{'='*60}")
    print(f"  RAG-MIA — {LLM_NAME} × {RAG_NAME}")
    print(f"  Prompt #{PROMPT_ID}: {ATTACK_PROMPTS[PROMPT_ID][:60]}...")
    print(f"{'='*60}")

    mia   = MembershipInferenceAttack(rag, prompt_id=PROMPT_ID)
    eval_ = mia.evaluate(members, non_members, verbose=True)

    summary    = eval_.summary()
    total_cost = sum(r.cost_usd for r in eval_.results)

    print(f"\n--- Résultats ---")
    print(f"  TPR        : {summary['tpr']:.4f}   (% membres correctement détectés)")
    print(f"  FPR        : {summary['fpr']:.4f}   (% non-membres faussement classés membres)")
    print(f"  AUC-ROC    : {summary['auc_roc']:.4f}")
    print(f"  Manquants  : {summary['n_missing']} / {summary['n_total']}")
    print(f"  Coût total : ${total_cost:.6f}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"mia_{LLM_NAME}_{RAG_NAME}_prompt{PROMPT_ID}"):
        mlflow.log_params({
            "attack":        "membership_inference",
            "llm":           LLM_NAME,
            "rag":           RAG_NAME,
            "prompt_id":     PROMPT_ID,
            "n_members":     N_MEMBERS,
            "n_non_members": N_NON_MEMBERS,
        })
        mlflow.log_metrics({
            "mia_tpr":        summary["tpr"],
            "mia_fpr":        summary["fpr"],
            "mia_auc_roc":    summary["auc_roc"],
            "mia_n_missing":  float(summary["n_missing"]),
            "cost_usd_total": total_cost,
        })

    return summary


def main():
    print("📥 Chargement du dataset...")
    all_docs = load_raw_documents()

    print("📦 Connexion à ChromaDB...")
    store = ChromaStore()

    print("🔍 Séparation membres / non-membres...")
    members, non_members = get_member_non_member_texts(store, all_docs)

    if len(members) < N_MEMBERS or len(non_members) < N_NON_MEMBERS:
        print("⚠️  Pas assez de documents. Réduis N_MEMBERS / N_NON_MEMBERS ou indexe plus de données.")
        return

    print("🤖 Instanciation Mistral + GraphRAG...")
    llm = MistralLLM()
    rag = GraphRAG(store, llm)

    try:
        summary = run_mia(rag, members, non_members)
    finally:
        rag.close()

    print(f"\n{'='*70}")
    print("  RÉSUMÉ")
    print(f"{'='*70}")
    print(f"{'LLM × RAG':<35} {'TPR':>8} {'FPR':>8} {'AUC-ROC':>10}")
    print("-" * 65)
    key = f"{LLM_NAME} × {RAG_NAME}"
    print(f"{key:<35} {summary['tpr']:>8.4f} {summary['fpr']:>8.4f} {summary['auc_roc']:>10.4f}")


if __name__ == "__main__":
    main()
