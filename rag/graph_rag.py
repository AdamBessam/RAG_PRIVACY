# knowledge_graph/graph_rag.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import spacy
from neo4j import GraphDatabase
from vectorstore.chroma_store import ChromaStore
from llms.base_llm import BaseLLM, LLMResponse
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    TOP_K, SENSITIVE_LABELS, SENSITIVE_ENTITY_TYPES,
    SPACY_MODEL,
)

from knowledge_graph.graph_builder import GraphBuilder
class GraphRAG:
    """
    GraphRAG — Knowledge Graph + Retrieval.

    Pipeline 5 étapes :
    1. Extraction entités de la query (spaCy)
    2. Match entités dans Neo4j
    3. Graph traversal 1-2 hops (Cypher)
    4. Reranking dense (ChromaDB embeddings)
    5. Génération LLM
    """

    # Paramètres
    N_ENTITY_MATCH  = 10   # max entités matchées dans Neo4j
    N_HOPS          = 2    # profondeur traversal
    N_CANDIDATES    = 30   # chunks candidats avant reranking
    KP              = TOP_K  # chunks finaux après reranking

    def __init__(self, store: ChromaStore, llm: BaseLLM):
        self.store  = store
        self.llm    = llm
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # Charger spaCy pour extraction entités
        try:
            self.nlp = spacy.load(SPACY_MODEL)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL])
            self.nlp = spacy.load(SPACY_MODEL)

        print("✅ GraphRAG initialisé")

    def close(self):
        self.driver.close()

    # ============================================================
    #  ÉTAPE 1 — Extraction entités de la query
    # ============================================================
    def _extract_query_entities(self, query: str) -> list[str]:
        """
        Extrait les entités et keywords importants de la query.
        Utilise spaCy NER + tokens importants.
        """
        doc = self.nlp(query)

        entities = []

        # Entités nommées spaCy
        for ent in doc.ents:
            entities.append(ent.text.strip())

        # Tokens importants (noms, adjectifs) si pas assez d'entités
        if len(entities) < 2:
            for token in doc:
                if (token.pos_ in ("NOUN", "PROPN", "ADJ")
                        and not token.is_stop
                        and len(token.text) > 3):
                    entities.append(token.text.strip())

        # Dédupliquer
        seen = set()
        result = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                result.append(e)

        return result[:10]  # max 10 entités

    # ============================================================
    #  ÉTAPE 2 — Match entités dans Neo4j
    # ============================================================
    def _match_entities_in_graph(
        self, query_entities: list[str]
    ) -> list[dict]:
        """
        Cherche les entités de la query dans le graphe Neo4j.
        Matching partiel insensible à la casse.
        Priorise les entités sensibles.
        """
        if not query_entities:
            return []

        matched = []

        with self.driver.session() as session:
            for entity_text in query_entities:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.text) CONTAINS toLower($text)
                       OR toLower($text) CONTAINS toLower(e.text)
                    RETURN e.text AS text,
                           e.type AS type,
                           e.sensitivity AS sensitivity
                    ORDER BY
                        CASE e.sensitivity
                            WHEN 'HEALTH'   THEN 0
                            WHEN 'SEX'      THEN 1
                            WHEN 'ETHNIC'   THEN 2
                            WHEN 'POLITICS' THEN 3
                            WHEN 'BELIEF'   THEN 4
                            ELSE 5
                        END,
                        size(e.text) DESC
                    LIMIT $limit
                """, {
                    "text":  entity_text,
                    "limit": self.N_ENTITY_MATCH,
                })

                for record in result:
                    matched.append({
                        "text":        record["text"],
                        "type":        record["type"],
                        "sensitivity": record["sensitivity"],
                    })

        # Dédupliquer
        seen = set()
        unique = []
        for e in matched:
            key = (e["text"], e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    # ============================================================
    #  ÉTAPE 3 — Graph Traversal
    # ============================================================
    def _traverse_graph(self, matched_entities: list[dict]) -> list[str]:
        """
        Traversal du graphe — récupère les chunk_ids pertinents.

        Pour chaque entité matchée :
        - Chunks directs qui MENTIONS cette entité
        - Entités voisines via CO_OCCURS_WITH (1-2 hops)
        - Chunks de ces entités voisines

        Retourne les chunk_ids triés par pertinence.
        """
        if not matched_entities:
            return []

        chunk_scores = {}  # chunk_id → score

        with self.driver.session() as session:
            for ent in matched_entities:

                # Boost sensibilité
                sensitivity_boost = (
                    2.0 if ent["sensitivity"] in SENSITIVE_LABELS else 1.0
                )

                # Chunks directs — score élevé
                result = session.run("""
                    MATCH (e:Entity {text: $text, type: $type})
                    MATCH (c:Chunk)-[:MENTIONS]->(e)
                    RETURN c.chunk_id AS chunk_id
                    LIMIT 10
                """, {
                    "text": ent["text"],
                    "type": ent["type"],
                })
                for record in result:
                    cid = record["chunk_id"]
                    chunk_scores[cid] = chunk_scores.get(cid, 0) + (
                        3.0 * sensitivity_boost
                    )

                # Voisins 1 hop via CO_OCCURS_WITH
                result = session.run("""
                    MATCH (e:Entity {text: $text, type: $type})
                    MATCH (e)-[r:CO_OCCURS_WITH]-(neighbor:Entity)
                    WHERE neighbor.sensitivity IN $sensitive_labels
                       OR $include_all = true
                    MATCH (c:Chunk)-[:MENTIONS]->(neighbor)
                    RETURN c.chunk_id AS chunk_id,
                           r.weight   AS weight,
                           neighbor.sensitivity AS sensitivity
                    ORDER BY r.weight DESC
                    LIMIT 20
                """, {
                    "text":           ent["text"],
                    "type":           ent["type"],
                    "sensitive_labels": list(SENSITIVE_LABELS),
                    "include_all":    len(matched_entities) < 3,
                })
                for record in result:
                    cid    = record["chunk_id"]
                    weight = record["weight"] or 1
                    n_boost = (
                        2.0 if record["sensitivity"] in SENSITIVE_LABELS
                        else 1.0
                    )
                    chunk_scores[cid] = chunk_scores.get(cid, 0) + (
                        weight * n_boost * sensitivity_boost
                    )

                # Voisins 2 hops — score plus faible
                if self.N_HOPS >= 2:
                    result = session.run("""
                        MATCH (e:Entity {text: $text, type: $type})
                        MATCH (e)-[:CO_OCCURS_WITH*2]-(neighbor2:Entity)
                        WHERE neighbor2.sensitivity IN $sensitive_labels
                        MATCH (c:Chunk)-[:MENTIONS]->(neighbor2)
                        RETURN c.chunk_id AS chunk_id
                        LIMIT 10
                    """, {
                        "text":             ent["text"],
                        "type":             ent["type"],
                        "sensitive_labels": list(SENSITIVE_LABELS),
                    })
                    for record in result:
                        cid = record["chunk_id"]
                        chunk_scores[cid] = chunk_scores.get(cid, 0) + (
                            0.5 * sensitivity_boost
                        )

        # Trier par score décroissant
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [cid for cid, _ in sorted_chunks[:self.N_CANDIDATES]]

    # ============================================================
    #  ÉTAPE 4 — Reranking dense
    # ============================================================
    def _rerank_dense(
        self, query: str, chunk_ids: list[str]
    ) -> list[dict]:
        """
        Reranking dense des chunks candidats via embeddings.
        Similaire au passage retriever on-the-fly de HHR.
        """
        if not chunk_ids:
            return []

        query_embedding = self.store.embedder.embed_single(query)

        # Récupérer embeddings des chunks candidats
        results = self.store.collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas", "embeddings"],
        )

        if not results["ids"]:
            return []

        passages = []
        for i in range(len(results["ids"])):
            passage_vec = np.array(results["embeddings"][i])
            similarity  = float(np.dot(query_embedding, passage_vec))

            passages.append({
                "chunk_id":       results["ids"][i],
                "text":           results["documents"][i],
                "similarity_score": similarity,
                "doc_id":         results["metadatas"][i]["doc_id"],
                "n_pii":          results["metadatas"][i].get("n_pii", 0),
                "pii_entities":   json.loads(
                    results["metadatas"][i].get("pii_entities", "[]")
                ),
            })

        # Trier par similarité
        passages.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Dédupliquer par doc_id
        seen_doc_ids = set()
        top_passages = []
        for p in passages:
            if p["doc_id"] not in seen_doc_ids:
                seen_doc_ids.add(p["doc_id"])
                top_passages.append(p)
            if len(top_passages) >= self.KP:
                break

        return top_passages

    # ============================================================
    #  FALLBACK — si graphe ne trouve rien
    # ============================================================
    def _fallback_naive(self, query: str) -> list[dict]:
        """
        Fallback vers ChromaDB direct si le graphe ne trouve rien.
        Garantit qu'on retourne toujours des chunks.
        """
        return self.store.query(query, top_k=self.KP)

    # ============================================================
    #  RETRIEVE — pipeline complet
    # ============================================================
    def retrieve(self, query: str) -> dict:
        """Pipeline GraphRAG complet — 4 étapes."""

        # Étape 1 — Extraction entités query
        query_entities = self._extract_query_entities(query)

        # Étape 2 — Match dans Neo4j
        matched_entities = self._match_entities_in_graph(query_entities)

        # Étape 3 — Traversal graphe
        candidate_chunk_ids = self._traverse_graph(matched_entities)

        # Étape 4 — Reranking dense
        if candidate_chunk_ids:
            top_chunks = self._rerank_dense(query, candidate_chunk_ids)
        else:
            top_chunks = self._fallback_naive(query)

        return {
            "chunks":            top_chunks,
            "query_entities":    query_entities,
            "matched_entities":  matched_entities,
            "n_candidates":      len(candidate_chunk_ids),
            "used_fallback":     len(candidate_chunk_ids) == 0,
        }

    # ============================================================
    #  GENERATE
    # ============================================================
    def generate(self, query: str, chunks: list[dict]) -> LLMResponse:
        prompt = self.llm.build_rag_prompt(query, chunks)
        return self.llm.generate(prompt)

    # ============================================================
    #  RUN — pipeline complet
    # ============================================================
    def run(self, query: str) -> dict:
        retrieval = self.retrieve(query)
        chunks    = retrieval["chunks"]
        result    = self.generate(query, chunks)

        return {
            "query":             query,
            "chunks":            chunks,
            "response":          result.response,
            "architecture":      "graph_rag",
            "llm":               result.llm_name,
            # GraphRAG spécifique
            "query_entities":    retrieval["query_entities"],
            "matched_entities":  retrieval["matched_entities"],
            "n_candidates":      retrieval["n_candidates"],
            "used_fallback":     retrieval["used_fallback"],
            # tokens
            "tokens_prompt":     result.tokens_prompt,
            "tokens_completion": result.tokens_completion,
            "tokens_total":      result.tokens_total,
            # coût
            "cost_usd":          result.cost_usd,
        }


if __name__ == "__main__":
    from llms.llama_llm import LlamaLLM
    from knowledge_graph.graph_builder import GraphBuilder

    # Construire le KG si pas encore fait
    builder = GraphBuilder()
    builder.build(force=False)
    builder.close()

    # Tester GraphRAG
    store = ChromaStore()
    llm   = LlamaLLM()
    rag   = GraphRAG(store=store, llm=llm)

    query  = "Who are the people mentioned in this document?"
    result = rag.run(query)

    print(f"\nQuery            : {query}")
    print(f"Architecture     : {result['architecture']}")
    print(f"Query entities   : {result['query_entities']}")
    print(f"Matched entities : {len(result['matched_entities'])}")
    print(f"Candidates       : {result['n_candidates']}")
    print(f"Fallback         : {result['used_fallback']}")
    print(f"Chunks final     : {len(result['chunks'])}")
    print(f"Réponse          : {result['response'][:200]}...")
    print(f"Tokens total     : {result['tokens_total']}")

    rag.close()