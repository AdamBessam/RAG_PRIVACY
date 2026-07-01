"""
recompute_cr_openai.py — Recalcule UNIQUEMENT le Context Recall (CR) sur les 300
requêtes d'un run déjà effectué, sans régénérer les réponses ni rappeler AE/PI.

Contexte : le harness calculait auparavant context_precision sous le nom "CR".
Zhang et al. (Table 2) mesurent en réalité RAGAS **context_recall**. Ce script
recalcule CR = context_recall proprement, sur les chunks BRUTS récupérés
(qualité de retrieval), et met à jour utility_scores.json en conservant les
SS/AR déjà calculés.

Coûts OpenAI :
  - embedding des 300 requêtes (re-retrieval des chunks bruts) : négligeable
  - juge GPT-4o pour context_recall seulement (pas SS ni AR)
  Aucune génération gpt-4o-mini, aucun AE, aucun PI.

Usage (depuis evaluation_zhang_openai/) :
  python recompute_cr_openai.py            # run v4 -> data/zhang_eval_openai_v2/
  python recompute_cr_openai.py --v3       # run v3 -> data/zhang_eval_openai/
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_zhang"))
sys.path.insert(0, str(Path(__file__).parent))

from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, TOP_K
from run_evaluation_openai import OpenAIZhangChromaStore  # réutilise l'index Chroma existant

ROOT = Path(__file__).parent.parent
SHARED_DATA_DIR = ROOT / "data" / "zhang_eval"


def build_raw_contexts(doc_index: dict, attacks: list[dict], top_k: int = TOP_K) -> list[list[str]]:
    """
    Re-récupère les chunks BRUTS par requête, SANS déduplication par document.

    store.query() ne garde qu'un chunk par doc_id (bon pour la diversité privacy,
    mais plafonne context_recall : le doc source n'apparaît qu'en 1 chunk sur ~6).
    Ici on interroge directement la collection Chroma → top_k chunks par
    similarité pure, donc plusieurs chunks du doc source → couverture réelle du
    retrieval, comparable au Naive RAG de Zhang et al. (Table 2).
    """
    store = OpenAIZhangChromaStore(doc_index)
    collection = store.collection
    embedder = store._embedder
    contexts: list[list[str]] = []
    for i, attack in enumerate(attacks):
        print(f"  retrieve [{i + 1}/{len(attacks)}] {attack['doc_id']}...", end="\r")
        q_emb = embedder.embed_single(attack["query"]).tolist()
        n_results = min(top_k, collection.count())
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=["documents"],
        )
        contexts.append(res["documents"][0])  # top_k chunks, non dédupliqués
    print()
    return contexts


def compute_context_recall(attacks: list[dict], contexts: list[list[str]], references: list[str]) -> float:
    """Calcule uniquement RAGAS context_recall (juge GPT-4o)."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_recall

    data = {
        "question":     [a["query"] for a in attacks],
        "contexts":     contexts,
        "ground_truth": references,
        # 'answer' non requis par context_recall mais fourni pour robustesse RAGAS
        "answer":       references,
    }
    dataset = Dataset.from_dict(data)

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0))
    context_recall.llm = llm

    result = evaluate(dataset, metrics=[context_recall])
    return float(result.get("context_recall", 0.0))


def main(use_v3: bool = False, top_k: int = TOP_K) -> None:
    data_dir = ROOT / "data" / ("zhang_eval_openai" if use_v3 else "zhang_eval_openai_v2")
    utility_path = data_dir / "utility_scores.json"
    run_label = "v3" if use_v3 else "v4"

    print(f"=== Recompute Context Recall — run {run_label} ({data_dir.name}), top_k={top_k} (non-dedup) ===\n")

    # 1. Données partagées
    with open(SHARED_DATA_DIR / "doc_index.json", encoding="utf-8") as f:
        doc_index = json.load(f)
    attacks = json.loads((SHARED_DATA_DIR / "attack_queries.json").read_text(encoding="utf-8"))
    print(f"1. {len(doc_index)} docs, {len(attacks)} requêtes")

    # 2. Réponses gold (cache réutilisé, aucune génération)
    from metric_utility import generate_reference_responses
    references = generate_reference_responses(attacks, doc_index)
    assert len(references) == len(attacks), "references/attacks length mismatch"

    # 3. Chunks bruts NON dédupliqués (re-retrieval, embedding requêtes seulement)
    print(f"2. Re-retrieval top-{top_k} chunks bruts, SANS dédup par doc...")
    contexts = build_raw_contexts(doc_index, attacks, top_k=top_k)

    # 4. Context Recall (juge GPT-4o, context_recall seulement)
    print("3. RAGAS context_recall (GPT-4o)...")
    cr = compute_context_recall(attacks, contexts, references)
    print(f"   CR (context_recall) = {cr:.4f}")

    # 5. Fusion dans utility_scores.json (conserve SS/AR existants)
    if utility_path.exists():
        with open(utility_path, encoding="utf-8") as f:
            utility = json.load(f)
        old_cr = utility.get("CR")
        utility["CR"] = cr
        with open(utility_path, "w", encoding="utf-8") as f:
            json.dump(utility, f, ensure_ascii=False, indent=2)
        print(f"\n4. utility_scores.json mis à jour : CR {old_cr} -> {cr:.4f} "
              f"(SS={utility.get('SS')}, AR={utility.get('AR')} conservés)")
    else:
        utility = {"CR": cr, "SS": None, "AR": None}
        with open(utility_path, "w", encoding="utf-8") as f:
            json.dump(utility, f, ensure_ascii=False, indent=2)
        print(f"\n4. utility_scores.json créé (SS/AR absents — relancer le harness pour les avoir)")

    print(f"\nDone. Model={OPENAI_EMBEDDING_MODEL} embedding, gpt-4o judge (CR only).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalcule Context Recall (CR) sur un run existant, CR-only.")
    parser.add_argument("--v3", action="store_true", help="Cible le run v3 (data/zhang_eval_openai/) au lieu de v4")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Nb de chunks non dédupliqués à récupérer (défaut {TOP_K})")
    args = parser.parse_args()
    main(use_v3=args.v3, top_k=args.top_k)
