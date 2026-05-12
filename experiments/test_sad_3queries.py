"""
Test SAD Detector sur 3 requêtes spécifiques — terminal only, pas de MLflow.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K
from countermeasure.cpb_naive_rag import CPBNaiveRAG
from llms.llama_llm import LlamaLLM
from rag.naive_rag import NaiveRAG
from vectorstore.chroma_store import ChromaStore

QUERIES = [
    "What title did she hold as Chairperson of the Wilhelmshaven/Friesland local branch of the party?",
]

def main():
    print("Loading ChromaStore...")
    store = ChromaStore()
    print("Loading Llama...")
    llm = LlamaLLM()
    cpb_rag = CPBNaiveRAG(naive_rag=NaiveRAG(store=store, llm=llm))

    for i, query in enumerate(QUERIES, 1):
        print("\n" + "=" * 70)
        print(f"[{i}] QUERY : {query}")
        print("=" * 70)

        result = cpb_rag.run(query, top_k=TOP_K)
        sad = result.get("cpb_sad_result")

        print(f"  Response     : {result['response'][:300]}")
        print(f"  SAD detected : {result['cpb_sad_detected']}")
        print(f"  Decision     : {result['cpb_sad_decision']}")
        print(f"  Categories   : {result['cpb_sad_categories']}")
        print(f"  Confidence   : {result['cpb_sad_confidence']:.2f}")
        print(f"  Max sim      : {(sad.max_similarity if sad else 0):.3f}")
        print(f"  Filter       : F{result['cpb_sad_filter']}")
        if sad:
            print(f"  Reasoning    : {sad.reasoning}")

if __name__ == "__main__":
    main()
