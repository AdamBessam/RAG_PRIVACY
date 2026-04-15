# knowledge_graph/graph_builder.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from tqdm import tqdm
from neo4j import GraphDatabase
from vectorstore.chroma_store import ChromaStore
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class GraphBuilder:
    """
    Construit le Knowledge Graph dans Neo4j depuis ChromaDB.
    Tourne une seule fois — vérifie si le KG existe déjà avant de construire.

    Structure du graphe :
    - Nœuds  : Document, Chunk, Entity
    - Relations : CONTAINS, MENTIONS, CO_OCCURS_WITH
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        print("✅ Connexion Neo4j établie")

    def close(self):
        self.driver.close()

    # ============================================================
    #  CHECK — KG déjà construit ?
    # ============================================================
    def is_built(self) -> bool:
        """Vérifie si le KG est déjà construit."""
        with self.driver.session() as session:
            result = session.run("MATCH (d:Document) RETURN count(d) AS n")
            count = result.single()["n"]
            return count > 0

    def get_stats(self) -> dict:
        """Retourne les statistiques du graphe."""
        with self.driver.session() as session:
            stats = {}
            for label in ["Document", "Chunk", "Entity"]:
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS n")
                stats[label] = r.single()["n"]
            for rel in ["CONTAINS", "MENTIONS", "CO_OCCURS_WITH"]:
                r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")
                stats[rel] = r.single()["n"]
            return stats

    # ============================================================
    #  CONTRAINTES ET INDEX
    # ============================================================
    def _create_constraints(self):
        """Crée les contraintes d'unicité et index pour les performances."""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT doc_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",

                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",

                "CREATE CONSTRAINT entity_key IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.text, e.type) IS UNIQUE",
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass

            # Index pour les recherches textuelles
            indexes = [
                "CREATE INDEX entity_text IF NOT EXISTS "
                "FOR (e:Entity) ON (e.text)",

                "CREATE INDEX entity_sensitivity IF NOT EXISTS "
                "FOR (e:Entity) ON (e.sensitivity)",

                "CREATE INDEX entity_type IF NOT EXISTS "
                "FOR (e:Entity) ON (e.type)",
            ]
            for index in indexes:
                try:
                    session.run(index)
                except Exception:
                    pass

        print("✅ Contraintes et index créés")

    # ============================================================
    #  ÉTAPE 1 — Nœuds Document
    # ============================================================
    def _build_documents(self, documents: list[dict]):
        """Crée les nœuds Document."""
        print(f"\n📥 Étape 1 — Création des nœuds Document ({len(documents)})...")

        with self.driver.session() as session:
            for doc in tqdm(documents, desc="Documents"):
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.year      = $year,
                        d.countries = $countries,
                        d.task      = $task
                """, {
                    "doc_id":    doc["doc_id"],
                    "year":      str(doc.get("year", "")),
                    "countries": str(doc.get("countries", "")),
                    "task":      str(doc.get("task", "")),
                })

        print(f"✅ {len(documents)} nœuds Document créés")

    # ============================================================
    #  ÉTAPE 2 — Nœuds Chunk + relations CONTAINS
    # ============================================================
    def _build_chunks(self, chunks: list[dict], batch_size: int = 200):
        """Crée les nœuds Chunk et les relations CONTAINS."""
        print(f"\n📥 Étape 2 — Création des nœuds Chunk ({len(chunks)})...")

        with self.driver.session() as session:
            for i in tqdm(
                range(0, len(chunks), batch_size),
                desc="Chunks"
            ):
                batch = chunks[i:i + batch_size]
                session.run("""
                    UNWIND $batch AS chunk
                    MERGE (c:Chunk {chunk_id: chunk.chunk_id})
                    SET c.text       = chunk.text,
                        c.char_start = chunk.char_start,
                        c.char_end   = chunk.char_end,
                        c.n_pii      = chunk.n_pii
                    WITH c, chunk
                    MATCH (d:Document {doc_id: chunk.doc_id})
                    MERGE (d)-[:CONTAINS]->(c)
                """, {"batch": [
                    {
                        "chunk_id":   c["chunk_id"],
                        "doc_id":     c["doc_id"],
                        "text":       c["text"][:500],
                        "char_start": c["char_start"],
                        "char_end":   c["char_end"],
                        "n_pii":      c.get("n_pii", 0),
                    }
                    for c in batch
                ]})

        print(f"✅ {len(chunks)} nœuds Chunk + relations CONTAINS créés")

    # ============================================================
    #  ÉTAPE 3 — Nœuds Entity + relations MENTIONS
    # ============================================================
    def _build_entities(self, chunks: list[dict], batch_size: int = 100):
        """
        Crée les nœuds Entity et les relations MENTIONS.
        MERGE sur (text, type) — une seule entité même si dans plusieurs chunks.
        """
        print(f"\n📥 Étape 3 — Création des nœuds Entity + relations MENTIONS...")

        total_entities = 0

        with self.driver.session() as session:
            for chunk in tqdm(chunks, desc="Entities"):
                pii = chunk.get("pii_entities", [])
                if isinstance(pii, str):
                    pii = json.loads(pii)

                if not pii:
                    continue

                for ent in pii:
                    if not ent["text"].strip():
                        continue

                    session.run("""
                        MERGE (e:Entity {text: $text, type: $type})
                        SET e.sensitivity = $sensitivity
                        WITH e
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, {
                        "text":        ent["text"].strip(),
                        "type":        ent.get("type", "UNKNOWN"),
                        "sensitivity": ent.get("sensitivity", "NOT_CONFIDENTIAL"),
                        "chunk_id":    chunk["chunk_id"],
                    })
                    total_entities += 1

        print(f"✅ {total_entities} relations MENTIONS créées")

    # ============================================================
    #  ÉTAPE 4 — Relations CO_OCCURS_WITH
    # ============================================================
    def _build_cooccurrences(self, chunks: list[dict]):
        """
        Crée les relations CO_OCCURS_WITH entre entités.
        Deux entités sont liées si elles apparaissent dans le même chunk.
        Poids = nombre de chunks où elles co-occurrent.
        """
        print(f"\n📥 Étape 4 — Création des relations CO_OCCURS_WITH...")

        total_cooc = 0

        with self.driver.session() as session:
            for chunk in tqdm(chunks, desc="Co-occurrences"):
                pii = chunk.get("pii_entities", [])
                if isinstance(pii, str):
                    pii = json.loads(pii)

                # Filtrer entités valides
                entities = [
                    e for e in pii
                    if e["text"].strip()
                ]

                if len(entities) < 2:
                    continue

                # Générer toutes les paires
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        e1 = entities[i]
                        e2 = entities[j]

                        if e1["text"].strip() == e2["text"].strip():
                            continue

                        # MERGE avec incrément du poids
                        session.run("""
                            MATCH (e1:Entity {text: $text1, type: $type1})
                            MATCH (e2:Entity {text: $text2, type: $type2})
                            MERGE (e1)-[r:CO_OCCURS_WITH]-(e2)
                            ON CREATE SET r.weight = 1
                            ON MATCH  SET r.weight = r.weight + 1
                        """, {
                            "text1": e1["text"].strip(),
                            "type1": e1.get("type", "UNKNOWN"),
                            "text2": e2["text"].strip(),
                            "type2": e2.get("type", "UNKNOWN"),
                        })
                        total_cooc += 1

        print(f"✅ {total_cooc} relations CO_OCCURS_WITH créées")

    # ============================================================
    #  BUILD — Pipeline complet
    # ============================================================
    def build(self, force: bool = False):
        """
        Construit le KG complet depuis ChromaDB.
        Si déjà construit → skip sauf si force=True.
        """
        if not force and self.is_built():
            stats = self.get_stats()
            print(f"✅ KG déjà construit — skip")
            print(f"   Documents  : {stats['Document']}")
            print(f"   Chunks     : {stats['Chunk']}")
            print(f"   Entities   : {stats['Entity']}")
            print(f"   CONTAINS   : {stats['CONTAINS']}")
            print(f"   MENTIONS   : {stats['MENTIONS']}")
            print(f"   CO_OCCURS  : {stats['CO_OCCURS_WITH']}")
            return

        print("🔨 Construction du Knowledge Graph...")

        # Créer contraintes et index
        self._create_constraints()

        # Charger depuis ChromaDB
        print("\n📥 Chargement des chunks depuis ChromaDB...")
        store   = ChromaStore()
        results = store.collection.get(
            include=["documents", "metadatas"]
        )

        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "chunk_id":    results["ids"][i],
                "doc_id":      results["metadatas"][i]["doc_id"],
                "text":        results["documents"][i],
                "char_start":  results["metadatas"][i].get("char_start", 0),
                "char_end":    results["metadatas"][i].get("char_end", 0),
                "n_pii":       results["metadatas"][i].get("n_pii", 0),
                "pii_entities": results["metadatas"][i].get("pii_entities", "[]"),
            })

        print(f"   {len(chunks)} chunks chargés")

        # Extraire les documents uniques
        seen_docs = set()
        documents = []
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                documents.append({
                    "doc_id":    doc_id,
                    "year":      "",
                    "countries": "",
                    "task":      "",
                })

        # Construire le graphe étape par étape
        self._build_documents(documents)
        self._build_chunks(chunks)
        self._build_entities(chunks)
        self._build_cooccurrences(chunks)

        # Stats finales
        stats = self.get_stats()
        print(f"\n✅ Knowledge Graph construit avec succès !")
        print(f"   Documents       : {stats['Document']}")
        print(f"   Chunks          : {stats['Chunk']}")
        print(f"   Entities        : {stats['Entity']}")
        print(f"   CONTAINS        : {stats['CONTAINS']}")
        print(f"   MENTIONS        : {stats['MENTIONS']}")
        print(f"   CO_OCCURS_WITH  : {stats['CO_OCCURS_WITH']}")


if __name__ == "__main__":
    builder = GraphBuilder()
    builder.build(force=False)
    builder.close()