"""
Debug pas-à-pas : trace question par question, étape par étape (B0..B7),
sur un sous-ensemble de 10 questions "target" + 10 questions "untarget".

Mêmes conditions que run_sage_eval.py (même pipeline CPBNaiveRAGV3, defense="cpb",
k=1, même corpus healthcaremagic), mais sur 20 requêtes au lieu de 500, avec le
détail de chaque brique pour comprendre où la fuite apparaît et si B6/B7 la
rattrapent réellement.

Ce script REJOUE cpb.run() étape par étape (mêmes appels publics : retrieve(),
generate(), sad_detector.detect(), response_guard.guard()) au lieu de l'appeler
en boîte noire, pour pouvoir capturer chaque résultat intermédiaire.

Usage :
  python sage_reference/debug_trace_10q.py

Sorties :
  - trace détaillée par question sur la console
  - résumé agrégé (B5 brut vs FINAL post-B7), comparable aux lignes de
    run_sage_eval.py
  - trace JSON complète : sage_reference/debug_trace_10q.json
"""
import json
from pathlib import Path

from run_sage_eval import K, _get_cpb_v3
from sage_metrics import evaluate_repeat, evaluate_rouge, extract_target_diseases

N = 10
HERE = Path(__file__).parent


# ======================================================================
# Reproduction des callbacks de reask (copiés de cpb_naive_rag_v3.run())
# ======================================================================
def _reask_factory(cpb, masked_query, chunks):
    def reask():
        strengthened = (
            masked_query
            + "\n\nPrivacy instruction: answer only from the masked context. "
            + "Do not reveal names, identifiers, locations, dates, or any raw personal data. "
            + "Keep placeholders exactly as provided."
        )
        return cpb.generate(strengthened, chunks).response
    return reask


def _sad_reask_factory(cpb, masked_query, chunks):
    def sad_reask(attribute_category: str) -> str:
        constrained = (
            masked_query
            + f"\n\nPrivacy instruction: answer this question without revealing "
            + f"any {attribute_category} information about any individual. "
            + "Do not mention specific medical conditions, diagnoses, "
            + "political beliefs, ethnic origins, sexual orientation, "
            + "or religious beliefs."
        )
        return cpb.generate(constrained, chunks).response
    return sad_reask


# ======================================================================
# Trace d'une seule requête, brique par brique
# ======================================================================
def trace_one(cpb, query: str, attack: str, disease: str | None) -> dict:
    step = {"attack": attack, "query": query, "disease": disease}

    # ── B1 + B2 (via retrieve()) ────────────────────────────────────────
    retrieval = cpb.retrieve(query, top_k=K)
    qr = retrieval["query_risk"]
    step["b1_query_risk_score"] = qr.score
    step["b1_signals"] = dict(qr.signals)
    step["b1_ner_entities"] = qr.ner_entities
    step["b2_decision"] = retrieval["decision"]
    step["b2_direct_suppression"] = cpb.budget_gate.direct_suppression(qr.score)
    step["b2_jailbreak_suppression"] = cpb.budget_gate.jailbreak_suppression(qr.signals)

    if retrieval["decision"] == "direct_suppression":
        msg = "I cannot process this request because it asks for sensitive context disclosure."
        step["raw_chunks"] = []
        step["b5_raw_response"] = msg
        step["final_response"] = msg
        step["leak_b5"] = {"repeat_prompt": 0, "target_info": 0}
        step["leak_final"] = {"repeat_prompt": 0, "target_info": 0, "rouge_prompt": 0}
        return step

    raw_chunks = [c["text"] for c in retrieval["raw_chunks"]]
    safe_chunks_meta = [
        {
            "text": c.get("text", ""),
            "cpb_pii_score": c.get("cpb_pii_score", 0.0),
            "cpb_n_replacements": c.get("cpb_n_replacements", 0),
            "similarity_score": raw.get("similarity_score"),
        }
        for c, raw in zip(retrieval["chunks"], retrieval["raw_chunks"])
    ]
    step["raw_chunks"] = raw_chunks
    step["b3b4_query_pii_score"] = retrieval.get("query_pii_score", 0.0)
    step["b3b4_query_pii_findings_count"] = retrieval.get("query_pii_findings_count", 0)
    step["b3b4_masked_query"] = retrieval.get("masked_query", query)
    step["b3b4_chunks"] = safe_chunks_meta

    masked_query = step["b3b4_masked_query"]
    chunks = retrieval["chunks"]

    # ── B5 : génération LLM brute (avant SAD/Guard) ─────────────────────
    llm_response = cpb.generate(masked_query, chunks)
    step["b5_raw_response"] = llm_response.response

    leak_b5 = evaluate_repeat(
        [llm_response.response], [raw_chunks[:K]],
        target_diseases=[disease] if disease is not None else None,
    )
    step["leak_b5"] = {
        "repeat_prompt": leak_b5["repeat_prompt"],
        "target_info": leak_b5["target_info"],
    }

    # ── B6 : SADDetectorV3 ───────────────────────────────────────────────
    sad = cpb.sad_detector.detect(
        query=masked_query,
        chunks=chunks,
        response=llm_response.response,
        reask_callback=_sad_reask_factory(cpb, masked_query, chunks),
    )
    step["b6_sad_detected"] = sad.sad_detected
    step["b6_decision"] = sad.decision
    step["b6_categories"] = sad.attribute_categories
    step["b6_max_similarity"] = sad.max_similarity
    step["b6_confidence"] = sad.confidence
    step["b6_filter_triggered"] = sad.filter_triggered
    step["b6_reasoning"] = sad.reasoning
    step["b6_response"] = sad.response

    # ── B7 : CPBResponseGuard ───────────────────────────────────────────
    guarded = cpb.response_guard.guard(
        response=sad.response,
        reask_callback=_reask_factory(cpb, masked_query, chunks),
    )
    step["b7_decision"] = guarded.decision
    step["b7_leakage_score"] = guarded.leakage_score
    step["b7_n_findings"] = guarded.n_findings
    step["b7_n_replacements"] = guarded.n_replacements
    step["b7_reason"] = guarded.reason
    step["final_response"] = guarded.response

    leak_final_rep = evaluate_repeat(
        [guarded.response], [raw_chunks[:K]],
        target_diseases=[disease] if disease is not None else None,
    )
    leak_final_rou = evaluate_rouge([guarded.response], [raw_chunks[:K]])
    step["leak_final"] = {
        "repeat_prompt": leak_final_rep["repeat_prompt"],
        "target_info": leak_final_rep["target_info"],
        "rouge_prompt": leak_final_rou["rouge_prompt"],
    }
    return step


# ======================================================================
# Affichage console (compact)
# ======================================================================
def _trunc(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def print_step(i: int, step: dict) -> None:
    print("=" * 100)
    print(f"[{step['attack']} #{i}] Q: {_trunc(step['query'], 140)}")
    sig = step["b1_signals"]
    sig_str = ", ".join(f"{k}={v:.2f}" for k, v in sig.items())
    print(f"  B1 risk={step['b1_query_risk_score']:.3f}  ({sig_str})")
    print(f"  B2 decision={step['b2_decision']}  "
          f"direct_suppr={step['b2_direct_suppression']}  jailbreak={step['b2_jailbreak_suppression']}")

    if step["b2_decision"] == "direct_suppression":
        print(f"  -> BLOQUÉ en B2 (BudgetGate). Réponse finale : {step['final_response']}")
        return

    print(f"  B3/B4 query_pii_score={step['b3b4_query_pii_score']:.3f} "
          f"findings={step['b3b4_query_pii_findings_count']}")
    for j, c in enumerate(step["b3b4_chunks"]):
        print(f"    chunk[{j}] sim={c['similarity_score']:.3f} pii_score={c['cpb_pii_score']:.3f} "
              f"n_repl={c['cpb_n_replacements']}")
        print(f"       raw : {_trunc(step['raw_chunks'][j])}")
        print(f"       safe: {_trunc(c['text'])}")

    print(f"  B5 raw_llm_response: {_trunc(step['b5_raw_response'])}")
    lb5 = step["leak_b5"]
    print(f"     -> leak_b5  : repeat={lb5['repeat_prompt']} target_info={lb5['target_info']}")

    print(f"  B6 SAD detected={step['b6_sad_detected']} decision={step['b6_decision']} "
          f"categories={step['b6_categories']} max_sim={step['b6_max_similarity']:.2f} "
          f"confidence={step['b6_confidence']:.2f} filter={step['b6_filter_triggered']}")
    print(f"     reasoning: {_trunc(step['b6_reasoning'], 160)}")
    print(f"     response après B6: {_trunc(step['b6_response'])}")

    print(f"  B7 Guard decision={step['b7_decision']} leakage_score={step['b7_leakage_score']:.3f} "
          f"n_findings={step['b7_n_findings']} n_replacements={step['b7_n_replacements']} "
          f"reason={step['b7_reason']}")
    print(f"  FINAL response: {_trunc(step['final_response'])}")

    lf = step["leak_final"]
    leaked = bool(lf["repeat_prompt"] or lf["rouge_prompt"])
    verdict = "FUITE NON RATTRAPÉE" if leaked else "fuite absente/rattrapée"
    print(f"     -> leak_final: repeat={lf['repeat_prompt']} target_info={lf['target_info']} "
          f"rouge={lf['rouge_prompt']}  [{verdict}]")


# ======================================================================
# Orchestration
# ======================================================================
if __name__ == "__main__":
    target_q = json.load(open(HERE / "questions" / "target-chatdoctor-question.json", encoding="utf-8"))[:N]
    untarget_q = json.load(open(HERE / "questions" / "untarget-chatdoctor-question.json", encoding="utf-8"))[:N]
    diseases = extract_target_diseases(target_q)

    cpb = _get_cpb_v3()
    br = cpb.bootstrap_result
    print("#" * 100)
    print("B0 BOOTSTRAP (calculé une seule fois, indépendamment des 20 requêtes ci-dessous)")
    print(f"  domain            = {br.domain}  (confidence={br.domain_confidence:.2f}, "
          f"used_fallback={br.used_fallback})")
    print(f"  learned_types     = {sorted(br.learned_types)}")
    print(f"  dynamic_categories= {br.dynamic_categories}")
    for cat, phrases in br.dynamic_taxonomy.items():
        print(f"    [{cat}] {len(phrases)} phrases d'ancrage, hints={sorted(br.category_hints.get(cat, []))}")
        for p in phrases[:2]:
            print(f"       ex: {_trunc(p, 120)}")

    all_steps = []
    for i, q in enumerate(target_q, start=1):
        step = trace_one(cpb, q, "target", diseases[i - 1])
        print_step(i, step)
        all_steps.append(step)

    for i, q in enumerate(untarget_q, start=1):
        step = trace_one(cpb, q, "untarget", None)
        print_step(i, step)
        all_steps.append(step)

    # ── Résumé agrégé : B5 brut (avant contre-mesure) vs FINAL (post B7) ───
    target_steps = [s for s in all_steps if s["attack"] == "target"]
    untarget_steps = [s for s in all_steps if s["attack"] == "untarget"]

    t_contexts = [s["raw_chunks"][:K] for s in target_steps]
    t_b5 = evaluate_repeat([s["b5_raw_response"] for s in target_steps], t_contexts, target_diseases=diseases)
    t_final = evaluate_repeat([s["final_response"] for s in target_steps], t_contexts, target_diseases=diseases)

    u_contexts = [s["raw_chunks"][:K] for s in untarget_steps]
    u_b5_rep = evaluate_repeat([s["b5_raw_response"] for s in untarget_steps], u_contexts)
    u_b5_rou = evaluate_rouge([s["b5_raw_response"] for s in untarget_steps], u_contexts)
    u_final_rep = evaluate_repeat([s["final_response"] for s in untarget_steps], u_contexts)
    u_final_rou = evaluate_rouge([s["final_response"] for s in untarget_steps], u_contexts)

    print("#" * 100)
    print("RESUME (10 target + 10 untarget) — comparaison AVANT (B5 brut) / APRES (FINAL post-B7)")
    print(f"  TARGET   B5    -> Target Info={t_b5['target_info']}  Repeat Prompts={t_b5['repeat_prompt']}")
    print(f"  TARGET   FINAL -> Target Info={t_final['target_info']}  Repeat Prompts={t_final['repeat_prompt']}")
    print(f"  UNTARGET B5    -> Repeat Prompt={u_b5_rep['repeat_prompt']} "
          f"Repeat Context={u_b5_rep['repeat_context']} ROUGE Prompt={u_b5_rou['rouge_prompt']} "
          f"ROUGE Context={u_b5_rou['rouge_context']}")
    print(f"  UNTARGET FINAL -> Repeat Prompt={u_final_rep['repeat_prompt']} "
          f"Repeat Context={u_final_rep['repeat_context']} ROUGE Prompt={u_final_rou['rouge_prompt']} "
          f"ROUGE Context={u_final_rou['rouge_context']}")

    summary = {
        "target_b5": t_b5,
        "target_final": t_final,
        "untarget_b5_repeat": u_b5_rep,
        "untarget_b5_rouge": u_b5_rou,
        "untarget_final_repeat": u_final_rep,
        "untarget_final_rouge": u_final_rou,
    }

    out_path = HERE / "debug_trace_10q.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "bootstrap": {
                    "domain": br.domain,
                    "domain_confidence": br.domain_confidence,
                    "used_fallback": br.used_fallback,
                    "learned_types": sorted(br.learned_types),
                    "dynamic_categories": br.dynamic_categories,
                    "dynamic_taxonomy": br.dynamic_taxonomy,
                },
                "summary": summary,
                "steps": all_steps,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nTrace complète écrite dans {out_path}")
