# experiments/run_membership_inference.py
"""
Lance le RAG-MIA (black-box) sur les LLMs locaux Llama et Mistral via Ollama,
pour les 4 architectures RAG : Naive, HHR, Self, Graph.

Usage
-----
python experiments/run_membership_inference.py

Résultats loggés dans MLflow : tpr, fpr, auc_roc, n_missing, cost_usd_total
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random

import mlflow

from vectorstore.chroma_store import ChromaStore
from llms.llama_llm import LlamaLLM
from llms.mistral_llm import MistralLLM
from rag.naive_rag import NaiveRAG
from rag.hhr_rag import HHRRAG
from rag.self_rag import SelfRAG
from rag.graph_rag import GraphRAG
from attacks.membership_inference import MembershipInferenceAttack, ATTACK_PROMPTS
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from data.loader import load_raw_documents


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_MEMBERS     = 50   # documents présents dans la base à tester
N_NON_MEMBERS = 50   # documents absents à tester
PROMPT_ID     = 2    # meilleur prompt selon le paper (Table 1)
SEED          = 42


def get_member_non_member_texts(store: ChromaStore, all_docs: list[dict]) -> tuple[list[str], list[str]]:
    """
    Membres   : docs du split train indexés dans ChromaDB (identifiés via doc_id).
    Non-membres : docs du split validation, jamais indexés.
    """
    from datasets import load_dataset
    from config import DATASET_NAME

    # --- membres : train indexé ---
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

    # --- non-membres : split validation (jamais indexé) ---
    val_dataset = load_dataset(DATASET_NAME, split="validation")
    val_docs    = list(val_dataset)
    random.shuffle(val_docs)
    non_members = [d["text"].strip() for d in val_docs[:N_NON_MEMBERS]]

    print(f"✅ Membres trouvés     : {len(members)}")
    print(f"✅ Non-membres trouvés : {len(non_members)}")
    return members, non_members


def run_mia(llm_name: str, rag_name: str, rag, members: list[str], non_members: list[str]):
    print(f"\n{'='*60}")
    print(f"  RAG-MIA — {llm_name} × {rag_name}")
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
    with mlflow.start_run(run_name=f"mia_{llm_name}_{rag_name}_prompt{PROMPT_ID}"):
        mlflow.log_params({
            "attack":        "membership_inference",
            "llm":           llm_name,
            "rag":           rag_name,
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


def build_rag_instances(store: ChromaStore, llm) -> list[tuple[str, object]]:
    """Retourne les architectures RAG instanciées avec le même store et llm."""
    instances = [
        ("naive_rag", NaiveRAG(store, llm)),
        ("hhr_rag",   HHRRAG(store, llm)),
        ("self_rag",  SelfRAG(store, llm)),
    ]
    try:
        instances.append(("graph_rag", GraphRAG(store, llm)))
    except Exception as e:
        print(f"⚠️  GraphRAG ignoré (Neo4j indisponible) : {e}")
    return instances


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

    llms = [
        ("llama3.1:8b", LlamaLLM()),
        ("mistral:7b",  MistralLLM()),
    ]

    all_results = {}  # {(llm_name, rag_name): summary}

    for llm_name, llm in llms:
        for rag_name, rag in build_rag_instances(store, llm):
            key = f"{llm_name} × {rag_name}"
            all_results[key] = run_mia(llm_name, rag_name, rag, members, non_members)
            # Fermer la connexion Neo4j si GraphRAG
            if hasattr(rag, "close"):
                rag.close()

    print(f"\n{'='*70}")
    print("  RÉSUMÉ COMPARATIF")
    print(f"{'='*70}")
    print(f"{'LLM × RAG':<35} {'TPR':>8} {'FPR':>8} {'AUC-ROC':>10}")
    print("-" * 65)
    for key, s in all_results.items():
        print(f"{key:<35} {s['tpr']:>8.4f} {s['fpr']:>8.4f} {s['auc_roc']:>10.4f}")


if __name__ == "__main__":
    main()
