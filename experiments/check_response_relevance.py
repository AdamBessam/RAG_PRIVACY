# experiments/check_response_relevance.py
"""
Script simple pour vérifier si les réponses sont pertinentes par rapport aux questions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.naive_rag import NaiveRAG
from vectorstore.chroma_store import ChromaStore
from llms.llama_llm import LlamaLLM
from embeddings.embedder import Embedder
from data.query_generator import load_queries
import numpy as np


def check_relevance():
    """Vérifie la pertinence des 5 premières réponses."""
    
    print("\n" + "="*100)
    print("  VÉRIFICATION DE LA PERTINENCE DES RÉPONSES")
    print("="*100)
    
    # Charger le RAG
    print("\n📥 Chargement du RAG...")
    store = ChromaStore()
    llm = LlamaLLM()
    rag = NaiveRAG(store=store, llm=llm)
    
    # Charger l'embedder (pour calculer la similarité)
    embedder = Embedder()
    
    # Charger les requêtes
    print("📥 Chargement des requêtes...")
    queries = load_queries()[:5]  # Teste sur 5 requêtes
    print(f"   {len(queries)} requêtes chargées\n")
    
    # Tester chaque requête
    for i, q in enumerate(queries, 1):
        print(f"\n{'─'*100}")
        print(f"QUESTION [{i}]:")
        print(f"  {q['query']}")
        print(f"{'─'*100}")
        
        # Générer la réponse via RAG
        result = rag.run(q["query"])
        response = result["response"]
        
        print(f"\nRÉPONSE GÉNÉRÉE:")
        print(f"  {response}")
        
        # Calculer la pertinence (similarité cosinus)
        q_embedding = embedder.embed_single(q["query"])
        r_embedding = embedder.embed_single(response)
        similarity = np.dot(q_embedding, r_embedding)
        
        # Afficher le score
        print(f"\n📊 SCORE DE PERTINENCE: {similarity:.4f}")
        
        # Interprétation
        if similarity >= 0.7:
            status = "✅ TRÈS PERTINENT"
        elif similarity >= 0.5:
            status = "⚠️  PARTIELLEMENT PERTINENT"
        else:
            status = "❌ PAS PERTINENT"
        
        print(f"   {status}")
        
        print(f"\nDÉTAILS:")
        print(f"  Tokens utilisés    : {result['tokens_total']}")
        print(f"  Chunks récupérés   : {len(result['chunks'])}")
        
    print(f"\n\n{'='*100}")
    print("  INTERPRÉTATION")
    print("="*100)
    print("""
0.0 - 0.3  : ❌ Pas pertinent du tout
0.3 - 0.5  : ⚠️  Peu pertinent
0.5 - 0.7  : ✓ Plutôt pertinent
0.7 - 0.9  : ✅ Très pertinent
0.9 - 1.0  : ✅✅ Extrêmement pertinent
    """)


if __name__ == "__main__":
    check_relevance()
