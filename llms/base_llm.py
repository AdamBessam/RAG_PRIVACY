# llms/base_llm.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from config import compute_cost, QUERY_LOG_MAX_CHARS


@dataclass
class LLMResponse:
    """Réponse standardisée pour tous les LLMs."""
    response:          str
    tokens_prompt:     int
    tokens_completion: int
    tokens_total:      int
    cost_usd:          float
    llm_name:          str


class BaseLLM(ABC):
    """
    Interface commune à tous les LLMs du benchmark.
    Chaque LLM doit implémenter generate().
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """
        Génère une réponse à partir d'un prompt.
        Doit retourner un LLMResponse avec tokens et coût.
        """
        pass

    def build_rag_prompt(self, question: str, chunks: list[dict]) -> str:
        """
        Construit le prompt RAG standard :
        contexte (chunks récupérés) + question.
        """
        context = "\n\n---\n\n".join([c["text"] for c in chunks])
        return (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    def build_cost(self, tokens_prompt: int, tokens_completion: int) -> float:
        """Calcule le coût USD de la requête."""
        return compute_cost(self.name, tokens_prompt, tokens_completion)

    def truncate_for_log(self, text: str) -> str:
        """Tronque un texte pour le logging MLflow."""
        return text[:QUERY_LOG_MAX_CHARS]