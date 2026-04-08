# rag/naive_rag.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.base_llm import BaseLLM, LLMResponse
from config import TOP_K


class NaiveRAG:
    """
    Architecture RAG baseline.
    Retrieval top-k cosinus direct + génération LLM.
    Pas de reranking, pas de filtrage.
    C'est le point de référence absolu du benchmark.
    """

    def __init__(self, store: ChromaStore, llm: BaseLLM):
        self.store = store
        self.llm = llm

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Récupère les top_k chunks les plus similaires."""
        return self.store.query(query, top_k=top_k)

    def generate(self, query: str, chunks: list[dict]) -> LLMResponse:
        """Génère une réponse à partir des chunks récupérés."""
        prompt = self.llm.build_rag_prompt(query, chunks)
        return self.llm.generate(prompt)

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Pipeline complète : retrieve + generate.
        Retourne tout : chunks récupérés + réponse + tokens + coût.
        """
        chunks   = self.retrieve(query, top_k)
        result   = self.generate(query, chunks)

        return {
            "query":              query,
            "chunks":             chunks,
            "response":           result.response,
            "architecture":       "naive_rag",
            "llm":                result.llm_name,
            # tokens
            "tokens_prompt":      result.tokens_prompt,
            "tokens_completion":  result.tokens_completion,
            "tokens_total":       result.tokens_total,
            # coût
            "cost_usd":           result.cost_usd,
        }