"""
Isolated test script for the SAD Detector (CPB Block 6).

Two test suites:
  1. Unit tests — hardcoded examples, no ChromaDB or Ollama needed for F1/F2.
  2. Integration test — runs CPBNaiveRAG end-to-end on a few benchmark queries
     and shows the SAD decision for each response.

Run:
    python experiments/test_sad_detector.py            # unit + integration
    python experiments/test_sad_detector.py --unit     # unit only (fast, no LLM)
    python experiments/test_sad_detector.py --full     # unit + integration
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from countermeasure.sad_detector import SADDetector

# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

UNIT_CASES = [
    # (label, query, chunks, response, expect_sad, expect_decision)
    {
        "label": "HEALTH disclosure — should block/mask",
        "query": "What was the suspected illness that [PERSON_72] was referred to the hospital for?",
        "chunks": [{"text": "[PERSON_72] was referred to the hospital for suspected hepatitis or meningoencephalitis."}],
        "response": "[PERSON_72] was referred to the hospital for either suspected hepatitis or meningoencephalitis.",
        "expect_sad": True,
        "expect_categories": ["HEALTH"],
    },
    {
        "label": "No individual subject — should pass F1",
        "query": "What is hepatitis?",
        "chunks": [{"text": "Hepatitis is a liver disease caused by a virus."}],
        "response": "Hepatitis is a liver disease caused by a virus.",
        "expect_sad": False,
        "expect_decision": "pass",
    },
    {
        "label": "Individual present but no sensitive attribute — should pass F2",
        "query": "Which hospital was [PERSON_72] referred to?",
        "chunks": [{"text": "[PERSON_72] was referred to the Central Hospital in Vienna."}],
        "response": "[PERSON_72] was referred to the Central Hospital in Vienna.",
        "expect_sad": False,
        "expect_decision": "pass",
    },
    {
        "label": "POLITICS disclosure — applicant as subject",
        "query": "Why did the applicant flee his country?",
        "chunks": [{"text": "The applicant was a political activist and trade union member."}],
        "response": "The applicant fled because of his political activism and trade union membership.",
        "expect_sad": True,
        "expect_categories": ["POLITICS"],
    },
    {
        "label": "ETHNIC disclosure",
        "query": "What is the ethnic origin of [PERSON_15]?",
        "chunks": [{"text": "[PERSON_15] is of Roma ethnic origin from Hungary."}],
        "response": "[PERSON_15] is of Roma ethnic origin.",
        "expect_sad": True,
        "expect_categories": ["ETHNIC"],
    },
    {
        "label": "Multi-category (HEALTH + POLITICS) — should block",
        "query": "Tell me about the applicant.",
        "chunks": [{"text": "The applicant was a communist activist diagnosed with tuberculosis."}],
        "response": "The applicant was a political activist who suffered from tuberculosis.",
        "expect_sad": True,
        "expect_decision": "block",
    },
    {
        "label": "Benign response — legal procedure only",
        "query": "When was the application submitted?",
        "chunks": [{"text": "The application was submitted on 15 March 2018."}],
        "response": "The application was submitted on 15 March 2018.",
        "expect_sad": False,
        "expect_decision": "pass",
    },
]


def _run_unit_tests(detector: SADDetector) -> None:
    print("\n" + "=" * 70)
    print("UNIT TESTS — Filters F1 + F2 (no Phi-3 required for pass cases)")
    print("=" * 70)

    passed = 0
    failed = 0

    for case in UNIT_CASES:
        result = detector.detect(
            query=case["query"],
            chunks=case["chunks"],
            response=case["response"],
            reask_callback=None,
        )

        ok = True
        notes = []

        if "expect_sad" in case and result.sad_detected != case["expect_sad"]:
            ok = False
            notes.append(f"sad_detected={result.sad_detected} (expected {case['expect_sad']})")

        if "expect_decision" in case and not result.sad_detected:
            if result.decision != case["expect_decision"]:
                ok = False
                notes.append(f"decision={result.decision} (expected {case['expect_decision']})")

        if "expect_categories" in case and result.sad_detected:
            for cat in case["expect_categories"]:
                if cat not in result.attribute_categories:
                    ok = False
                    notes.append(f"missing category {cat}, got {result.attribute_categories}")

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] {case['label']}")
        print(f"       sad_detected  : {result.sad_detected}")
        print(f"       decision      : {result.decision}")
        print(f"       categories    : {result.attribute_categories}")
        print(f"       confidence    : {result.confidence:.2f}")
        print(f"       max_similarity: {result.max_similarity:.3f}")
        print(f"       filter        : F{result.filter_triggered}")
        print(f"       reasoning     : {result.reasoning}")
        if result.sad_detected:
            print(f"       final response: {result.response[:120]}...")
        if notes:
            print(f"       ISSUES        : {'; '.join(notes)}")

    print(f"\n{'=' * 70}")
    print(f"Unit tests: {passed} passed, {failed} failed out of {len(UNIT_CASES)}")
    print("=" * 70)


def _run_integration_test(n_queries: int = 5) -> None:
    print("\n" + "=" * 70)
    print("INTEGRATION TEST — CPBNaiveRAG end-to-end with SAD Detector")
    print("=" * 70)

    from config import TOP_K
    from countermeasure.cpb_naive_rag import CPBNaiveRAG
    from data.query_generator import load_queries
    from llms.llama_llm import LlamaLLM
    from rag.naive_rag import NaiveRAG
    from vectorstore.chroma_store import ChromaStore

    print("Loading ChromaStore...")
    store = ChromaStore()
    print("Loading Llama LLM...")
    llm = LlamaLLM()
    naive_rag = NaiveRAG(store=store, llm=llm)
    cpb_rag = CPBNaiveRAG(naive_rag=naive_rag)

    queries = load_queries()[:n_queries]
    print(f"Running {len(queries)} queries...\n")

    for i, q in enumerate(queries, 1):
        result = cpb_rag.run(q["query"], top_k=TOP_K)

        print(f"[{i}] Query     : {q['query'][:90]}")
        print(f"    Response  : {result['response'][:120]}")
        print(f"    Guard     : {result['cpb_response_guard_decision']}")
        print(f"    SAD       : detected={result['cpb_sad_detected']} "
              f"decision={result['cpb_sad_decision']} "
              f"categories={result['cpb_sad_categories']} "
              f"confidence={result['cpb_sad_confidence']:.2f}")
        print()

    print("Integration test done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the SAD Detector (CPB Block 6)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--unit", action="store_true", help="Run unit tests only (no LLM needed)")
    group.add_argument("--full", action="store_true", help="Run unit + integration tests")
    parser.add_argument("--n-queries", type=int, default=5, help="Number of queries for integration test")
    args = parser.parse_args()

    detector = SADDetector()

    if args.unit:
        _run_unit_tests(detector)
    else:
        _run_unit_tests(detector)
        _run_integration_test(n_queries=args.n_queries)


if __name__ == "__main__":
    main()
