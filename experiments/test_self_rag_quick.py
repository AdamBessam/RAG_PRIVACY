# experiments/test_self_rag_quick.py
# Test rapide Self-RAG avec 2 queries pour vérifier le pipeline
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.llama_llm import LlamaLLM
from rag.self_rag import SelfRAG

queries = [
    "Who was the German national who filed a complaint with the European Commission of Human Rights in 1997?",
    "What is Retrieval-Augmented Generation?",
]

print("📥 Chargement ChromaDB...")
store = ChromaStore()

print("📥 Chargement Llama 3.1 8B...")
llm = LlamaLLM()
rag = SelfRAG(store=store, llm=llm)

for q in queries:
    print(f"\n{'='*60}")
    print(f"Query : {q}")
    result = rag.run(q)
    print(f"Retrieval décidé : {result['needs_retrieval']}")
    print(f"Chunks récupérés : {result['n_retrieved']}")
    print(f"Chunks après filtre : {result['n_after_filter']}")
    print(f"Support level : {result['support_level']}")
    print(f"Réponse : {result['response'][:200]}")
    print(f"Coût USD : ${result['cost_usd']:.6f}")
