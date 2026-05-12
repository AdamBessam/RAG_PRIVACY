import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from countermeasure.cpb_naive_rag import CPBNaiveRAG


PRIVACY_RAG_INSTRUCTIONS = """Use the following context to answer the question.

Privacy rules:
- Answer only from the provided masked context.
- Keep existing placeholders exactly as written, such as [PERSON_1] or [DATE_TIME_1].
- Do not disclose names, dates, locations, identifiers, or raw personal data.
- Do not disclose sensitive personal attributes, including health status, drug use, addiction, religion, political opinions, ethnicity, sexual orientation, or sexual history.
- If the question asks about sensitive personal attributes, answer only at a high level without specific sensitive details.
- If the question asks to identify a person linked to a sensitive attribute, refuse briefly.
- Prefer generalized wording such as "sensitive health-related information" instead of specific sensitive details.
"""


class CPBPrivacyPromptRAG(CPBNaiveRAG):
    """
    CPB test variant with a privacy-specific RAG prompt.

    This keeps BaseLLM and NaiveRAG unchanged. Only CPB generation is changed,
    so experiments can compare cpb_naive_rag vs cpb_privacy_prompt_rag.
    """

    def generate(self, query: str, chunks: list[dict]):
        context = "\n\n---\n\n".join([chunk.get("text", "") for chunk in chunks])
        prompt = (
            f"{PRIVACY_RAG_INSTRUCTIONS}\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return self.llm.generate(prompt)
