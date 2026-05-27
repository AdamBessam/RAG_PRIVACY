"""
Étape 5 — Test ciblé de la correction Medical NER (scispaCy).

Teste uniquement les 61 queries qui avaient encore des fuites PII dans CPB
(leaked_cpb_residual.csv) SANS re-lancer le LLM — on vérifie au niveau des
chunks anonymisés si les entités médicales (DEM) sont maintenant masquées.

Logique :
  1. Pour chaque query du fichier residual, on lance CPB.retrieve()
     → récupère raw_chunks (bruts) + safe_chunks (anonymisés par le nouveau CPB)
  2. On identifie les entités DEM dans les raw_chunks (ground-truth du dataset)
  3. On vérifie si ces termes apparaissent encore dans le texte des safe_chunks
  4. Si absent → le Medical NER a masqué la fuite ✅
  5. Si encore présent → autre type de fuite (non médical) ⚠️

Usage:
    python test_contre_mesure_ildpiltest/05_test_medical_fix.py
"""
import csv
import json
import sys
import sqlite3
from pathlib import Path

# pysqlite3 n'est pas disponible sur Windows — on réutilise sqlite3 natif
if "pysqlite3" not in sys.modules:
    sys.modules["pysqlite3"] = sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from test_contre_mesure_ildpiltest.config import (
    CHROMA_DIR, COLLECTION_NAME, TOP_K,
)
from test_contre_mesure_ildpiltest._store import IldpilTestStore
from countermeasure.cpb_pii import PresidioPIIAnalyzer, PresidioPIIAnonymizer, BudgetGate
from countermeasure.cpb_query_risk import QueryRiskScorer
from rag.naive_rag import NaiveRAG
from countermeasure.cpb_naive_rag import CPBNaiveRAG

RESIDUAL_CSV = Path(__file__).parent / "leaked_cpb_residual.csv"
RESULTS_OUT  = Path(__file__).parent / "medical_fix_results.csv"

# Types considérés comme médicaux dans le dataset ildpil
MEDICAL_ENTITY_TYPE = "DEM"


def load_residual_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_dem_entities_from_chunks(chunks: list[dict]) -> list[dict]:
    """Extrait toutes les entités DEM (médicales) des chunks ground-truth."""
    dem_entities = []
    for chunk in chunks:
        for ent in chunk.get("pii_entities", []):
            if ent.get("type") == MEDICAL_ENTITY_TYPE:
                text = ent.get("text", "").strip()
                if len(text) > 2:
                    dem_entities.append({
                        "text":        text,
                        "chunk_id":    chunk.get("chunk_id", ""),
                        "sensitivity": ent.get("sensitivity", ""),
                    })
    return dem_entities


def check_still_visible(dem_entities: list[dict], safe_chunks: list[dict]) -> list[dict]:
    """
    Vérifie quelles entités DEM sont encore visibles dans les chunks anonymisés.
    Retourne la liste des entités encore présentes (= non masquées).
    """
    safe_text = " ".join(c.get("text", "") for c in safe_chunks).lower()
    still_visible = []
    for ent in dem_entities:
        if ent["text"].lower() in safe_text:
            still_visible.append(ent)
    return still_visible


def main():
    print("=" * 60)
    print("  TEST CORRECTION MEDICAL NER — CPB v2 (scispaCy)")
    print("=" * 60)

    if not RESIDUAL_CSV.exists():
        print(f"ERREUR : {RESIDUAL_CSV} introuvable.")
        print("Lance d'abord 03_run_benchmark.py pour générer le CSV.")
        sys.exit(1)

    # ── Initialisation ────────────────────────────────────────────────────────
    print("\nInitialisation du store ChromaDB...")
    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)

    print("Initialisation NaiveRAG + CPBNaiveRAG (avec Medical NER)...")
    # LLM fictif — on ne génère pas de réponse, juste retrieve()
    class _DummyLLM:
        name = "dummy"
        def generate(self, *a, **kw): return None

    class _DummyNaiveRAG:
        def __init__(self, store):
            self.store = store
            self.llm   = _DummyLLM()
        def retrieve(self, query, top_k=TOP_K):
            return self.store.query(query, top_k=top_k)
        def generate(self, *a, **kw): return None

    naive_rag = _DummyNaiveRAG(store)
    cpb = CPBNaiveRAG(naive_rag=naive_rag)

    # Vérifie que scispaCy est actif
    if not cpb.pii_analyzer.medical_recognizer.is_available():
        print("\n⚠ ATTENTION : scispaCy non disponible — le test ne peut pas valider la correction.")
        print("  Installe : pip install scispacy")
        print("  Puis     : pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")
        sys.exit(1)

    # ── Chargement des 61 queries résiduelles ─────────────────────────────────
    rows = load_residual_queries(RESIDUAL_CSV)
    print(f"\n{len(rows)} queries résiduelles chargées depuis {RESIDUAL_CSV.name}\n")

    # ── Test par query ────────────────────────────────────────────────────────
    results = []
    n_had_dem          = 0   # queries avec entités DEM dans les chunks
    n_dem_fully_fixed  = 0   # queries où TOUTES les entités DEM sont maintenant masquées
    n_dem_partial_fix  = 0   # queries où certaines entités DEM sont masquées
    n_no_dem           = 0   # queries sans entités DEM (fuite d'un autre type)
    all_masked_terms   = []
    all_remaining_terms = []

    for row in tqdm(rows, desc="Test queries"):
        query    = row["query"]
        query_id = row.get("query_id", "?")

        # Retrieve + anonymise (NEW CPB avec scispaCy)
        retrieval   = cpb.retrieve(query, top_k=TOP_K)
        raw_chunks  = retrieval.get("raw_chunks", [])
        safe_chunks = retrieval.get("chunks", [])

        # Entités DEM ground-truth dans les raw_chunks
        dem_entities = get_dem_entities_from_chunks(raw_chunks)

        if not dem_entities:
            n_no_dem += 1
            results.append({
                "query_id":       query_id,
                "query_type":     row.get("query_type", ""),
                "had_dem":        False,
                "dem_count":      0,
                "still_visible":  0,
                "masked_count":   0,
                "fix_status":     "no_dem_entity",
                "remaining":      "",
                "masked_terms":   "",
                "original_cpb_leaked": row.get("pii_leaked", ""),
            })
            continue

        n_had_dem += 1
        still_visible = check_still_visible(dem_entities, safe_chunks)
        n_masked = len(dem_entities) - len(still_visible)

        masked_terms    = [e["text"] for e in dem_entities if e not in still_visible]
        remaining_terms = [e["text"] for e in still_visible]

        all_masked_terms.extend(masked_terms)
        all_remaining_terms.extend(remaining_terms)

        if len(still_visible) == 0:
            fix_status = "fully_fixed"
            n_dem_fully_fixed += 1
        elif n_masked > 0:
            fix_status = "partially_fixed"
            n_dem_partial_fix += 1
        else:
            fix_status = "not_fixed"

        results.append({
            "query_id":            query_id,
            "query_type":          row.get("query_type", ""),
            "had_dem":             True,
            "dem_count":           len(dem_entities),
            "still_visible":       len(still_visible),
            "masked_count":        n_masked,
            "fix_status":          fix_status,
            "remaining":           " | ".join(remaining_terms),
            "masked_terms":        " | ".join(masked_terms),
            "original_cpb_leaked": row.get("pii_leaked", ""),
        })

    # ── Sauvegarde CSV ────────────────────────────────────────────────────────
    fieldnames = ["query_id", "query_type", "had_dem", "dem_count",
                  "masked_count", "still_visible", "fix_status",
                  "masked_terms", "remaining", "original_cpb_leaked"]
    with open(RESULTS_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Rapport ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RAPPORT — {len(rows)} queries testées")
    print(f"{'=' * 60}")
    print(f"  Queries avec entités DEM (médicales) : {n_had_dem}/{len(rows)}")
    print(f"  Queries sans entités DEM              : {n_no_dem}/{len(rows)}")
    print(f"  {'-' * 56}")
    print(f"  Correction TOTALE  (toutes DEM masquées) : {n_dem_fully_fixed}/{n_had_dem}")
    print(f"  Correction PARTIELLE (certaines DEM)     : {n_dem_partial_fix}/{n_had_dem}")
    print(f"  Non corrigé (fuite autre type)            : {n_had_dem - n_dem_fully_fixed - n_dem_partial_fix}/{n_had_dem}")

    if all_masked_terms:
        from collections import Counter
        top_masked = Counter(all_masked_terms).most_common(10)
        print(f"\n  Top termes DEM maintenant masqués :")
        for term, count in top_masked:
            print(f"    [{count}x] {term}")

    if all_remaining_terms:
        top_remaining = Counter(all_remaining_terms).most_common(10)
        print(f"\n  Termes DEM encore visibles (non couverts par scispaCy) :")
        for term, count in top_remaining:
            print(f"    [{count}x] {term}")

    print(f"\n  Résultats détaillés : {RESULTS_OUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
