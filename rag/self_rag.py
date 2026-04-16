# rag/self_rag.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorstore.chroma_store import ChromaStore
from llms.base_llm import BaseLLM, LLMResponse
from config import TOP_K


class SelfRAG:
    """
    Self-RAG simulé par prompting (Asai et al., 2023).

    Justification : Le vrai Self-RAG nécessite un LLM fine-tuné sur 150k exemples
    avec des tokens de réflexion générés par GPT-4 (4x A100 80GB). Pour garantir
    un contrôle expérimental rigoureux — la seule variable étant l'architecture de
    retrieval — nous simulons les mécanismes via prompting sur les mêmes LLMs
    utilisés dans toutes les architectures du benchmark.

    Pipeline 3 étapes :
    1. RETRIEVE decision  — Le LLM décide si un retrieval est nécessaire
    2. ISREL filtering    — Le LLM filtre les chunks non pertinents
    3. Generate + ISSUP   — Génération avec auto-évaluation du support
    """

    RETRIEVE_THRESHOLD = 0.5   # seuil de décision retrieval (simulé)

    def __init__(self, store: ChromaStore, llm: BaseLLM):
        self.store = store
        self.llm   = llm

    # ============================================================
    #  ÉTAPE 1 — Décision de retrieval
    # ============================================================
    def _decide_retrieve(self, query: str) -> tuple[bool, LLMResponse]:
        """
        Simule le token [Retrieve] du papier.
        Demande au LLM si la query nécessite une recherche factuelle.
        Retourne (needs_retrieval, llm_response).
        """
        prompt = (
            "You are evaluating whether a query requires searching an external "
            "document database to be answered correctly.\n\n"
            "Answer with ONLY 'yes' or 'no'.\n"
            "- 'yes' if the query asks for specific factual information about "
            "people, events, dates, or sensitive personal data.\n"
            "- 'no' if the query is general, hypothetical, or can be answered "
            "from common knowledge without specific documents.\n\n"
            f"Query: {query}\n\n"
            "Needs retrieval (yes/no):"
        )
        response = self.llm.generate(prompt)
        needs_retrieval = "yes" in response.response.strip().lower()
        return needs_retrieval, response

    # ============================================================
    #  ÉTAPE 2 — Filtrage de pertinence (ISREL)
    # ============================================================
    def _filter_relevant_chunks(
        self, query: str, chunks: list[dict]
    ) -> tuple[list[dict], list[LLMResponse]]:
        """
        Simule le token [ISREL] du papier.
        Pour chaque chunk, le LLM évalue si il est pertinent pour la query.
        Retourne les chunks pertinents + les réponses LLM de filtrage.
        """
        relevant_chunks = []
        filter_responses = []

        for chunk in chunks:
            prompt = (
                "Evaluate if the following passage provides useful information "
                "to answer the query.\n\n"
                "Answer with ONLY 'relevant' or 'irrelevant'.\n\n"
                f"Query: {query}\n\n"
                f"Passage: {chunk['text'][:300]}\n\n"
                "Relevance (relevant/irrelevant):"
            )
            response = self.llm.generate(prompt)
            filter_responses.append(response)

            if "relevant" in response.response.strip().lower():
                relevant_chunks.append(chunk)

        # Si aucun chunk pertinent → garder le meilleur par similarité
        if not relevant_chunks and chunks:
            relevant_chunks = [chunks[0]]

        return relevant_chunks, filter_responses

    # ============================================================
    #  ÉTAPE 3 — Génération + auto-évaluation (ISSUP)
    # ============================================================
    def _generate_with_critique(
        self, query: str, chunks: list[dict]
    ) -> tuple[LLMResponse, str]:
        """
        Génère la réponse et simule le token [ISSUP] du papier.
        Retourne (réponse LLM, niveau de support : fully/partially/no_support).
        """
        # Génération
        prompt = self.llm.build_rag_prompt(query, chunks)
        response = self.llm.generate(prompt)

        # Auto-évaluation du support
        critique_prompt = (
            "Given the query, the context passages, and the generated answer, "
            "evaluate how well the answer is supported by the context.\n\n"
            "Answer with ONLY one of: 'fully_supported', 'partially_supported', "
            "'no_support'.\n\n"
            f"Query: {query}\n\n"
            f"Context: {chunks[0]['text'][:300] if chunks else 'none'}\n\n"
            f"Answer: {response.response[:200]}\n\n"
            "Support level (fully_supported/partially_supported/no_support):"
        )
        critique_response = self.llm.generate(critique_prompt)
        raw = critique_response.response.strip().lower()

        if "fully" in raw:
            support = "fully_supported"
        elif "partially" in raw:
            support = "partially_supported"
        else:
            support = "no_support"

        return response, support

    # ============================================================
    #  RETRIEVE
    # ============================================================
    def retrieve(self, query: str) -> dict:
        """
        Pipeline de retrieval Self-RAG :
        1. Décide si retrieval nécessaire
        2. Si oui : récupère + filtre par pertinence
        3. Si non : retourne liste vide
        """
        # Étape 1 — décision retrieval
        needs_retrieval, retrieve_response = self._decide_retrieve(query)

        tokens_retrieve = retrieve_response.tokens_total
        cost_retrieve   = retrieve_response.cost_usd

        if not needs_retrieval:
            return {
                "chunks":          [],
                "needs_retrieval": False,
                "n_retrieved":     0,
                "n_after_filter":  0,
                "support_level":   "no_support",
                "tokens_retrieve": tokens_retrieve,
                "cost_retrieve":   cost_retrieve,
                "tokens_filter":   0,
                "cost_filter":     0.0,
            }

        # Étape 2 — retrieval + filtrage ISREL
        raw_chunks = self.store.query(query, top_k=TOP_K)
        filtered_chunks, filter_responses = self._filter_relevant_chunks(
            query, raw_chunks
        )

        tokens_filter = sum(r.tokens_total for r in filter_responses)
        cost_filter   = sum(r.cost_usd for r in filter_responses)

        return {
            "chunks":          filtered_chunks,
            "needs_retrieval": True,
            "n_retrieved":     len(raw_chunks),
            "n_after_filter":  len(filtered_chunks),
            "support_level":   None,  # sera rempli après génération
            "tokens_retrieve": tokens_retrieve,
            "cost_retrieve":   cost_retrieve,
            "tokens_filter":   tokens_filter,
            "cost_filter":     cost_filter,
        }

    # ============================================================
    #  RUN — pipeline complet
    # ============================================================
    def run(self, query: str) -> dict:
        retrieval = self.retrieve(query)
        chunks    = retrieval["chunks"]

        if not chunks:
            # Pas de retrieval → génération directe sans contexte
            prompt   = f"Answer the following question:\n\n{query}\n\nAnswer:"
            result   = self.llm.generate(prompt)
            support  = "no_support"
        else:
            # Génération + auto-évaluation ISSUP
            result, support = self._generate_with_critique(query, chunks)

        # Coût total = retrieval + filtrage + génération
        total_tokens = (
            retrieval["tokens_retrieve"]
            + retrieval["tokens_filter"]
            + result.tokens_total
        )
        total_cost = (
            retrieval["cost_retrieve"]
            + retrieval["cost_filter"]
            + result.cost_usd
        )

        return {
            "query":             query,
            "chunks":            chunks,
            "response":          result.response,
            "architecture":      "self_rag",
            "llm":               result.llm_name,
            # Self-RAG spécifique
            "needs_retrieval":   retrieval["needs_retrieval"],
            "n_retrieved":       retrieval["n_retrieved"],
            "n_after_filter":    retrieval["n_after_filter"],
            "support_level":     support,
            # tokens (incluant les appels de réflexion)
            "tokens_prompt":     result.tokens_prompt,
            "tokens_completion": result.tokens_completion,
            "tokens_total":      total_tokens,
            # coût total
            "cost_usd":          total_cost,
        }
