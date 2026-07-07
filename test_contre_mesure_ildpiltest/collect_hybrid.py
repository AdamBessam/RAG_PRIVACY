"""
Collecte de réponses SÉCURISÉES par TYPE de question — récupération HybridRAG
(dense + BM25 + RRF) protégée par la contre-mesure CPB v5 (masquage sélectif
domain-aware). Pour constituer un jeu de données de démonstration à montrer à
l'encadrant.

Chaîne : question ─► HybridRAG (retrieval) ─► CPB v5 (défense) ─► réponse.
Le HybridRAG sert UNIQUEMENT à récupérer les chunks ; la contre-mesure CPB v5
enveloppe ce RAG (naive_rag=hybrid) et sécurise la réponse (masquage de la
requête, masquage sélectif des PII dans les chunks, garde SAD/response-guard).

Pour chaque type (normal, direct, injection, dgea, mia) on prend N questions,
et pour chacune on enregistre :
  - la question d'origine,
  - la requête après masquage CPB (cpb_masked_query),
  - les doc_id récupérés (avant masquage, avec score dense ou tag bm25),
  - les signaux défensifs CPB (risque requête, décision SAD, response-guard),
  - la réponse FINALE sécurisée.

Sorties (dans test_contre_mesure_ildpiltest/) :
  - reponses_hybrid_cpbv5_par_type.md    : lisible, groupé par type (encadrant)
  - reponses_hybrid_cpbv5_par_type.json  : structuré, réexploitable

Usage :
  python test_contre_mesure_ildpiltest/collect_hybrid.py                       # 20 par type, tous les types
  python test_contre_mesure_ildpiltest/collect_hybrid.py --per-type 10
  python test_contre_mesure_ildpiltest/collect_hybrid.py --types normal,injection --per-type 20
  python test_contre_mesure_ildpiltest/collect_hybrid.py --mask-min-weight 0.5 --no-domain-hints
  python test_contre_mesure_ildpiltest/collect_hybrid.py --out mon_fichier.md
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_contre_mesure_ildpiltest.config import (
    BENCHMARK_DIR, CHROMA_DIR, COLLECTION_NAME, QUERIES_FILE, TOP_K,
)
from test_contre_mesure_ildpiltest._store import IldpilTestStore

ALL_TYPES = ["normal", "direct", "injection", "dgea", "mia"]

TYPE_LABEL = {
    "normal":    "Questions légitimes (contenu)",
    "direct":    "Extraction ciblée de PII (IKEA-style)",
    "injection": "Prompt injection",
    "dgea":      "Jailbreak autoritaire (DGEA-style)",
    "mia":       "Membership inference (MIA)",
}


def sep(title: str = "") -> None:
    print("=" * 100)
    if title:
        print(f"  {title}")
        print("=" * 100)


def docs_line(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        sim = c.get("similarity_score")
        tag = f"{sim:.3f}" if isinstance(sim, (int, float)) else "bm25"
        parts.append(f"{c.get('doc_id')}({tag})")
    return "  ".join(parts) if parts else "(aucun)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=20,
                        help="Nombre de questions par type (défaut 20).")
    parser.add_argument("--types", default=",".join(ALL_TYPES),
                        help="Types à échantillonner, séparés par des virgules (défaut : tous).")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--mask-min-weight", type=float, default=0.5,
                        help="Curseur CPB v5 : masque une entité si poids(type) >= seuil (défaut 0.5).")
    parser.add_argument("--no-domain-hints", action="store_true",
                        help="Désactive le Signal 2 CPB v5 (category_hints de B0).")
    parser.add_argument("--out", default=str(BENCHMARK_DIR / "reponses_hybrid_cpbv5_par_type.md"),
                        help="Chemin du fichier Markdown de sortie.")
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    for t in types:
        if t not in ALL_TYPES:
            sys.exit(f"ERREUR : type inconnu '{t}'. Types valides : {ALL_TYPES}")

    # Sélection des questions par type
    if not QUERIES_FILE.exists():
        sys.exit(f"ERREUR : {QUERIES_FILE} introuvable.")
    with open(QUERIES_FILE, encoding="utf-8") as f:
        allq = json.load(f)

    selected: dict[str, list[dict]] = {}
    for t in types:
        pool = [q for q in allq if q.get("query_type") == t]
        selected[t] = pool[:args.per_type]
        print(f"  {t:10s} : {len(selected[t])} question(s) sélectionnée(s)")
    total = sum(len(v) for v in selected.values())
    print(f"Total : {total} question(s) — top_k={args.top_k}\n")

    # Imports lourds tardifs
    from llms.llama_llm import LlamaLLM
    from rag.hybrid_rag import HybridRAG
    from countermeasure_v5.cpb_naive_rag_v5 import CPBNaiveRAGV5

    print("Init ChromaDB + LLM + index BM25 + bootstrap CPB v5 (peut prendre 1-2 min)...")
    store = IldpilTestStore(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    if store.count() == 0:
        sys.exit("ERREUR : collection vide (lancez 01_index.py).")
    llm = LlamaLLM()

    # HybridRAG : uniquement la RÉCUPÉRATION des chunks (dense + BM25 + RRF).
    hybrid = HybridRAG(store=store, llm=llm)
    # CPB v5 : enveloppe le HybridRAG et SÉCURISE la réponse (masquage sélectif).
    cpb = CPBNaiveRAGV5(
        naive_rag=hybrid,
        mask_min_weight=args.mask_min_weight,
        use_domain_hints=not args.no_domain_hints,
    )

    records: list[dict] = []
    done = 0
    for t in types:
        sep(f"TYPE : {t}  —  {TYPE_LABEL.get(t, '')}")
        for q in selected[t]:
            query = str(q["query"])
            done += 1
            print(f"[{done}/{total}] ({t}) {q.get('query_id', '')}")
            print(f"Q: {query}")

            result = cpb.run(query, top_k=args.top_k)
            raw_chunks   = result.get("raw_chunks", [])
            resp         = result.get("response", "")
            masked_query = result.get("cpb_masked_query", query)
            risk = result.get("cpb_query_risk")
            sad_decision   = result.get("cpb_sad_decision", "pass")
            guard_decision = result.get("cpb_response_guard_decision", "pass")

            print(f"HYBRID docs (avant masquage) : {docs_line(raw_chunks)}")
            print(f"CPB  risque_requête={risk}  SAD={sad_decision}  guard={guard_decision}")
            print("--- Réponse SÉCURISÉE (HybridRAG + CPB v5) ---")
            print(resp)
            print()

            records.append({
                "query_id":       q.get("query_id"),
                "global_id":      q.get("global_id"),
                "query_type":     t,
                "query":          query,
                "cpb_masked_query": masked_query,
                "docs":           [
                    {"doc_id": c.get("doc_id"), "similarity_score": c.get("similarity_score")}
                    for c in raw_chunks
                ],
                "docs_line":      docs_line(raw_chunks),
                "cpb_query_risk":     risk,
                "cpb_sad_decision":   sad_decision,
                "cpb_response_guard": guard_decision,
                "response":       resp,
            })

    # --- Écriture JSON ---
    out_md = Path(args.out)
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Écriture Markdown (pour l'encadrant) ---
    lines: list[str] = []
    lines.append("# Réponses sécurisées par type de question — HybridRAG + CPB v5")
    lines.append("")
    lines.append(f"- Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- Récupération : HybridRAG (dense + BM25 + RRF) — LLM : Llama 3.1 8B")
    lines.append(f"- Contre-mesure : CPB v5 (masquage sélectif domain-aware)")
    lines.append(f"  - mask_min_weight = {args.mask_min_weight}")
    lines.append(f"  - domain_hints = {'OFF' if args.no_domain_hints else 'ON'}")
    lines.append(f"- top_k = {args.top_k}")
    lines.append(f"- Total : {total} question(s)")
    lines.append("")
    counts = {t: len(selected[t]) for t in types}
    lines.append("| Type | Description | Nb |")
    lines.append("|------|-------------|----|")
    for t in types:
        lines.append(f"| `{t}` | {TYPE_LABEL.get(t, '')} | {counts[t]} |")
    lines.append("")

    by_type: dict[str, list[dict]] = {t: [] for t in types}
    for r in records:
        by_type[r["query_type"]].append(r)

    for t in types:
        lines.append(f"## {t} — {TYPE_LABEL.get(t, '')}")
        lines.append("")
        for i, r in enumerate(by_type[t], 1):
            lines.append(f"### {i}. `{r['query_id']}`")
            lines.append("")
            lines.append(f"**Question d'origine :**")
            lines.append("")
            lines.append("> " + r["query"].replace("\n", "\n> "))
            lines.append("")
            if r["cpb_masked_query"] and r["cpb_masked_query"] != r["query"]:
                lines.append(f"**Requête après masquage CPB :**")
                lines.append("")
                lines.append("> " + str(r["cpb_masked_query"]).replace("\n", "\n> "))
                lines.append("")
            lines.append(f"**Documents récupérés (avant masquage) :** {r['docs_line']}")
            lines.append("")
            lines.append(
                f"**Défense CPB :** risque_requête = `{r['cpb_query_risk']}`, "
                f"SAD = `{r['cpb_sad_decision']}`, response-guard = `{r['cpb_response_guard']}`"
            )
            lines.append("")
            lines.append(f"**Réponse sécurisée :**")
            lines.append("")
            lines.append("> " + r["response"].replace("\n", "\n> "))
            lines.append("")
            lines.append("---")
            lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    sep("TERMINÉ")
    print(f"Markdown : {out_md}")
    print(f"JSON     : {out_json}")


if __name__ == "__main__":
    main()
