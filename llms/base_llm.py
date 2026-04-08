# llms/base_llm.py
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Interface commune à tous les LLMs du benchmark.
    Chaque LLM doit implémenter generate().
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Génère une réponse à partir d'un prompt."""
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