"""
Runner d'évaluation type SAGE sur HealthCareMagic, avec CPBNaiveRAGV3.

Pré-requis :
  - sage_metrics.py (dans le même dossier)
  - build_index.py exécuté au préalable (collection "healthcaremagic" indexée)
  - questions/target-chatdoctor-question.json
  - questions/untarget-chatdoctor-question.json
  - questions/per-chat-question.json + truth/per-chat-truth.json   (utilité)
  - pip install rouge_score

Peut être lancé depuis n'importe quel dossier (chemins résolus relativement au script) :
  python sage_reference/run_sage_eval.py
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

# Racine du projet — nécessaire pour importer countermeasure_v3, rag, llms, config
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rouge_score import rouge_scorer
from sage_metrics import evaluate_repeat, evaluate_rouge, extract_target_diseases

K = 1  # nombre de chunks récupérés (l'article utilise k=1)


# ======================================================================
# Singleton pipeline CPB v3 — instancié une seule fois (bootstrap coûteux)
# ======================================================================
_cpb_v3 = None


def _get_cpb_v3():
    global _cpb_v3
    if _cpb_v3 is None:
        from countermeasure_v3.cpb_naive_rag_v3 import CPBNaiveRAGV3
        from llms.llama_llm import LlamaLLM
        from rag.naive_rag import NaiveRAG
        from vectorstore.chroma_store import ChromaStore

        store     = ChromaStore(collection_name="healthcaremagic")
        llm       = LlamaLLM()
        naive_rag = NaiveRAG(store=store, llm=llm)
        _cpb_v3   = CPBNaiveRAGV3(naive_rag=naive_rag)
    return _cpb_v3


# ======================================================================
# 1) ADAPTATEUR
# ======================================================================
def run_system(query: str, defense: str):
    """
    Exécute une requête à travers CPB v3 et renvoie ce qu'il faut au scoring.

    defense == "cpb" -> pipeline complet B0..B7 (CPBNaiveRAGV3)

    Retourne (response, retrieved_chunks, sources) :
      response        : str        -> réponse FINALE après B6/B7
      retrieved_chunks: list[str]  -> chunks ORIGINAUX avant B4 (ground truth de fuite)
      sources         : list[str]  -> étiquettes de source (vide si indisponible)
    """
    if defense == "cpb":
        cpb    = _get_cpb_v3()
        result = cpb.run(query, top_k=K)

        # Réponse FINALE après B7 (ResponseGuard peut fix/reask/bloquer sad.response)
        guard_result = result.get("cpb_response_guard")
        response = guard_result.response if guard_result is not None else result["response"]

        # raw_chunks = chunks ORIGINAUX récupérés par NaiveRAG AVANT B4 (anonymisation)
        raw_chunks = [c["text"] for c in result.get("raw_chunks", [])]
        sources    = [""] * len(raw_chunks)
        return response, raw_chunks, sources

    raise ValueError(f"defense non reconnue : {defense!r}. Valeurs acceptées : 'cpb'")


# ======================================================================
# 2) Évaluation des attaques (targeted + untargeted)
# ======================================================================
def run_attack(attack: str, defense: str, queries):
    outputs, contexts, sources = [], [], []
    for q in queries:
        resp, chunks, srcs = run_system(q, defense)
        outputs.append(resp)
        contexts.append(chunks[:K])
        sources.append((srcs[:K]) if srcs else [""] * min(K, len(chunks)))

    diseases = extract_target_diseases(queries) if attack == "target" else None
    rep = evaluate_repeat(outputs, contexts, sources, target_diseases=diseases)

    row = {"attack": attack, "defense": defense}
    if attack == "target":
        row["Target Info"]    = rep["target_info"]
        row["Repeat Prompts"] = rep["repeat_prompt"]
    else:
        rou = evaluate_rouge(outputs, contexts, sources)
        row["Repeat Prompt"]  = rep["repeat_prompt"]
        row["Repeat Context"] = rep["repeat_context"]
        row["ROUGE Prompt"]   = rou["rouge_prompt"]
        row["ROUGE Context"]  = rou["rouge_context"]
    return row


# ======================================================================
# 3) Utilité (Table 1) : BLEU-1 et ROUGE-L vs réponse de référence
# ======================================================================
_TOK = re.compile(r"\w+")


def bleu1(candidate: str, reference: str) -> float:
    """BLEU-1 (précision unigramme x brevity penalty), formule Appendix A.8."""
    c = _TOK.findall(candidate.lower())
    r = _TOK.findall(reference.lower())
    if not c:
        return 0.0
    ref_counts  = Counter(r)
    cand_counts = Counter(c)
    clipped = sum(min(n, ref_counts.get(w, 0)) for w, n in cand_counts.items())
    p1 = clipped / len(c)
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / max(len(c), 1))
    return bp * p1


def run_utility(defense: str, questions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu_scores, rouge_scores = [], []
    for q, ref in zip(questions, references):
        resp, _, _ = run_system(q, defense)
        if not isinstance(resp, str):
            resp = ""
        bleu_scores.append(bleu1(resp, ref))
        rouge_scores.append(scorer.score(ref, resp)["rougeL"].fmeasure)
    n = max(len(bleu_scores), 1)
    return {
        "defense":  defense,
        "BLEU-1":   sum(bleu_scores) / n,
        "ROUGE-L":  sum(rouge_scores) / n,
    }


# ======================================================================
# 4) Orchestration
# ======================================================================
if __name__ == "__main__":
    # Chemins absolus par rapport à ce script — fonctionne quel que soit le dossier de lancement
    HERE = Path(__file__).parent
    target_q   = json.load(open(HERE / "questions" / "target-chatdoctor-question.json",   encoding="utf-8"))
    untarget_q = json.load(open(HERE / "questions" / "untarget-chatdoctor-question.json", encoding="utf-8"))
    # utilité (décommente quand prêt) :
    # perf_q   = json.load(open(HERE / "questions" / "per-chat-question.json", encoding="utf-8"))
    # perf_ref = json.load(open(HERE / "truth" / "per-chat-truth.json",        encoding="utf-8"))

    results = []
    for defense in ["cpb"]:
        results.append(run_attack("target",   defense, target_q))
        results.append(run_attack("untarget", defense, untarget_q))
        # results.append(run_utility(defense, perf_q, perf_ref))

    for r in results:
        print(r)
        # MLflow (décommente si un run est actif) :
        # import mlflow
        # mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, (int, float))})
