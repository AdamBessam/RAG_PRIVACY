"""
SAGE Pipeline — Génération offline des documents synthétiques.

Exécute Stage 1 (extraction attributs + génération) puis Stage 2 (agents Joe + Cathy)
sur tous les documents du split TEST de ildpil/text-anonymization-benchmark (555 docs).

Sortie : test_contre_mesure_ildpiltest/synthetic_docs.json

Usage :
    python sage/run_pipeline.py
    python sage/run_pipeline.py --protect-method sync    # Stage 1 seulement
    python sage/run_pipeline.py --protect-method agent2  # Stage 1 + Stage 2 (recommandé)
    python sage/run_pipeline.py --limit 10               # test rapide (10 docs)
    python sage/run_pipeline.py --resume                 # reprend depuis checkpoint

Modèles (conformes au papier) :
    Stage 1 : gpt-3.5-turbo
    Stage 2 : gpt-4
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from sage.doing_protect import get_synthetic_context, get_agent2_context

DATASET_NAME  = "ildpil/text-anonymization-benchmark"
DATASET_SPLIT = "test"
DATASET_NAME_KEY = "echr"   # clé de domaine pour les prompts SAGE

SENSITIVE_ENTITY_TYPES = ("PERSON", "DEM", "MISC", "ORG", "LOC")
NOT_SENSITIVE_LABEL    = "NOT_CONFIDENTIAL"
SENSITIVE_LABELS       = ("HEALTH", "POLITICS", "ETHNIC", "SEX", "BELIEF")

OUTPUT_FILE     = Path(__file__).parent.parent / "test_contre_mesure_ildpiltest" / "synthetic_docs.json"
CHECKPOINT_FILE = Path(__file__).parent.parent / "test_contre_mesure_ildpiltest" / "sage_checkpoint.json"


def get_sensitivity(entity_mentions: list) -> str:
    if not entity_mentions:
        return NOT_SENSITIVE_LABEL
    labels = set()
    for ent in entity_mentions:
        lbl = ent.get("confidential_status", NOT_SENSITIVE_LABEL)
        if isinstance(lbl, list):
            labels.update(lbl)
        else:
            labels.add(lbl)
    for sensitive in SENSITIVE_LABELS:
        if sensitive in labels:
            return sensitive
    return NOT_SENSITIVE_LABEL


def load_documents(dataset) -> list[dict]:
    """Charge et normalise les documents depuis HuggingFace."""
    documents = []
    for i, sample in enumerate(tqdm(dataset, desc="Chargement documents")):
        text = sample.get("text", "").strip()
        if not text:
            continue
        entity_mentions = sample.get("entity_mentions", []) or []
        pii_entities = []
        for ent in entity_mentions:
            ent_type = ent.get("entity_type", "")
            if ent_type not in SENSITIVE_ENTITY_TYPES:
                continue
            start = ent.get("start_offset", 0)
            end   = ent.get("end_offset", 0)
            pii_entities.append({
                "text":        ent.get("span_text", ""),
                "type":        ent_type,
                "start":       start,
                "end":         end,
                "sensitivity": ent.get("confidential_status", NOT_SENSITIVE_LABEL),
            })
        documents.append({
            "doc_id":       f"ildpil_test_{i:05d}",
            "text":         text,
            "pii_entities": pii_entities,
            "sensitivity":  get_sensitivity(entity_mentions),
        })
    return documents


def load_checkpoint() -> list[dict]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Checkpoint trouvé : {len(data)} docs déjà traités — reprise")
        return data
    return []


def save_checkpoint(results: list[dict]):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SAGE pipeline — génération docs synthétiques ECHR")
    parser.add_argument("--protect-method", default="agent2",
                        choices=["sync", "agent2"],
                        help="sync=Stage1 only | agent2=Stage1+Stage2 (recommandé, conforme au papier)")
    parser.add_argument("--attributes-llm", default="gpt-3.5-turbo",
                        choices=["gpt-3.5-turbo", "gpt-4"],
                        help="LLM Stage 1a — extraction attributs (papier: gpt-3.5-turbo)")
    parser.add_argument("--synthetic-llm", default="gpt-3.5-turbo",
                        choices=["gpt-3.5-turbo", "gpt-4"],
                        help="LLM Stage 1b — génération synthétique (papier: gpt-3.5-turbo)")
    parser.add_argument("--agents-llm", default="gpt-4",
                        choices=["gpt-3.5-turbo", "gpt-4"],
                        help="LLM Stage 2 — agents Joe+Cathy (papier: gpt-4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite le nombre de documents (test rapide)")
    parser.add_argument("--resume", action="store_true",
                        help="Reprend depuis le checkpoint existant")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERREUR : variable OPENAI_API_KEY non définie dans .env")
        sys.exit(1)

    print(f"Chargement du dataset : {DATASET_NAME} (split={DATASET_SPLIT})...")
    dataset   = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    documents = load_documents(dataset)

    if args.limit:
        documents = documents[:args.limit]
        print(f"Mode test : {args.limit} documents")

    print(f"{len(documents)} documents chargés\n")

    # ── Reprise depuis checkpoint ────────────────────────────────────────────
    checkpoint = load_checkpoint() if args.resume else []
    done_ids   = {r["doc_id"] for r in checkpoint}
    remaining  = [d for d in documents if d["doc_id"] not in done_ids]

    if not remaining:
        print("Tous les documents déjà traités (checkpoint complet).")
        results = checkpoint
    else:
        if checkpoint:
            print(f"{len(remaining)} documents restants sur {len(documents)} total\n")

        # Format attendu par SAGE : liste de listes (chaque doc = liste de chunks)
        # Ici : 1 chunk par doc (le texte complet, tronqué à 2000 chars pour le coût)
        ori_contexts = [[d["text"][:2000]] for d in remaining]

        # ── Stage 1 — génération synthétique ────────────────────────────────
        print(f"Stage 1 — extraction attributs + génération synthétique")
        print(f"  LLM attributs  : {args.attributes_llm}")
        print(f"  LLM synthèse   : {args.synthetic_llm}\n")

        _, synthetic_contexts = get_synthetic_context(
            ori_contexts,
            dataset=DATASET_NAME_KEY,
            attributes_llm=args.attributes_llm,
            synthetic_llm=args.synthetic_llm,
        )

        # ── Stage 2 — raffinement agents Joe + Cathy ────────────────────────
        if args.protect_method == "agent2":
            print(f"\nStage 2 — raffinement agents Joe + Cathy")
            print(f"  LLM agents : {args.agents_llm}\n")
            final_contexts = get_agent2_context(
                ori_contexts,
                synthetic_contexts,
                agents_llm=args.agents_llm,
            )
        else:
            # sync : Stage 1 seulement
            final_contexts = synthetic_contexts

        # ── Assemblage des résultats ─────────────────────────────────────────
        new_results = []
        for doc, synthetic_list in zip(remaining, final_contexts):
            # final_contexts[i] = liste de chunks synthétiques → on prend le premier
            synthetic_text = synthetic_list[0] if synthetic_list else doc["text"]
            new_results.append({
                "doc_id":         doc["doc_id"],
                "synthetic_text": synthetic_text,
                "sensitivity":    doc["sensitivity"],
                # pii_entities vide → texte synthétique ne contient pas de vraie PII
                "pii_entities":   [],
            })
            save_checkpoint(checkpoint + new_results)

        results = checkpoint + new_results

    # Nettoyage checkpoint
    CHECKPOINT_FILE.unlink(missing_ok=True)

    # ── Sauvegarde finale ────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  SAGE terminé : {len(results)} documents synthétiques")
    print(f"  Méthode      : {args.protect_method}")
    print(f"  Sortie       : {OUTPUT_FILE}")
    print(f"{'='*55}")
    print(f"\nEtape suivante : python test_contre_mesure_ildpiltest/04_index_sage.py")


if __name__ == "__main__":
    main()
