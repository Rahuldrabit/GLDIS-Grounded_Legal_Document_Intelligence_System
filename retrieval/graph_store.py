"""
Stage 2 — Neo4j Graph Store for GraphRAG Knowledge Indexing

Manages a Neo4j graph of:
  - Document nodes
  - Chunk nodes (keyed by chunk_id for bridge with FAISS)
  - Entity nodes (extracted by rule_based and ner extractors)
  - Relationships: IN_DOCUMENT, MENTIONS, CO_OCCURS_WITH, etc.

This layer sits between retrieval (FAISS/BM25) and generation, enabling
semantic expansion via graph neighborhood traversal.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from core.config import get_settings
from core.schemas import Chunk, EvidenceChunk, ExtractedField, StructuredExtraction

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Neo4j Connection (lazy-loaded singleton)
# ──────────────────────────────────────────────────────────────────────────────

_NEO4J_DRIVER = None


def _get_neo4j_driver():
    """Lazily load and cache a Neo4j driver."""
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is not None:
        return _NEO4J_DRIVER

    settings = get_settings()
    if not settings.neo4j_enabled:
        logger.info("Neo4j disabled; graph operations will be no-ops.")
        return None

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        _NEO4J_DRIVER = driver
        logger.info(f"Neo4j connected: {settings.neo4j_uri}")
        return driver
    except Exception as exc:
        logger.warning(f"Neo4j connection failed: {exc}. Graph operations disabled.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Graph Store Interface
# ──────────────────────────────────────────────────────────────────────────────

class GraphStore:
    """
    Manages indexing and retrieval from Neo4j for GraphRAG (Stage 2).
    Falls back gracefully to no-ops if Neo4j is unavailable.
    """

    def __init__(self):
        self.driver = _get_neo4j_driver()
        self.enabled = self.driver is not None

    # ── Index operations ─────────────────────────────────────────────────────

    def index_document(self, document_id: str, filename: str) -> bool:
        """Create or update a Document node."""
        if not self.enabled:
            return True  # silent no-op

        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (d:Document {id: $doc_id})
                    SET d.filename = $filename, d.updated_at = datetime()
                    """,
                    doc_id=document_id,
                    filename=filename,
                )
            logger.debug(f"Indexed document {document_id}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to index document {document_id}: {exc}")
            return False

    def index_chunks(self, chunks: List[Chunk], document_id: str) -> int:
        """
        Index a list of chunks as nodes and create IN_DOCUMENT edges.
        Returns number of chunks indexed.
        """
        if not self.enabled or not chunks:
            return len(chunks)

        indexed = 0
        try:
            with self.driver.session() as session:
                for chunk in chunks:
                    session.run(
                        """
                        MERGE (c:Chunk {id: $chunk_id})
                        SET c.document_id = $document_id, c.text = $text, c.page = $page,
                            c.section = $section, c.token_count = $token_count,
                            c.updated_at = datetime()
                        WITH c
                        MATCH (d:Document {id: $document_id})
                        MERGE (c)-[:IN_DOCUMENT]->(d)
                        """,
                        chunk_id=chunk.chunk_id,
                        document_id=document_id,
                        text=chunk.text,
                        page=chunk.page,
                        section=chunk.section,
                        token_count=chunk.token_count,
                    )
                    indexed += 1
            logger.debug(f"Indexed {indexed} chunks for document {document_id}")
        except Exception as exc:
            logger.warning(f"Failed to index chunks for {document_id}: {exc}")

        return indexed

    def index_entities(
        self,
        document_id: str,
        chunks: List[Chunk],
        extracted_fields: List[ExtractedField],
    ) -> int:
        """
        Index entities and create MENTIONS edges from chunks to entities,
        and CO_OCCURS_WITH edges between entities in the same chunk.
        Returns number of entity relationships created.
        """
        if not self.enabled or not extracted_fields:
            return 0

        try:
            chunk_lookup = {c.chunk_id: c for c in chunks}
            indexed = 0

            with self.driver.session() as session:
                # Create entity nodes
                entity_key_set: Set[Tuple[str, str]] = set()
                for field in extracted_fields:
                    entity_type = field.field.upper()
                    entity_value = str(field.value)[:256]  # Truncate long values
                    entity_key = (entity_type, entity_value)

                    if entity_key in entity_key_set:
                        continue
                    entity_key_set.add(entity_key)

                    session.run(
                        """
                        MERGE (e:Entity {type: $type, value: $value})
                        SET e.document_id = $document_id, e.updated_at = datetime()
                        """,
                        type=entity_type,
                        value=entity_value,
                        document_id=document_id,
                    )
                    indexed += 1

                # Create MENTIONS edges
                for field in extracted_fields:
                    entity_type = field.field.upper()
                    entity_value = str(field.value)[:256]
                    chunk_id = field.source_chunk_id

                    if chunk_id and chunk_id in chunk_lookup:
                        session.run(
                            """
                            MATCH (c:Chunk {id: $chunk_id})
                            MATCH (e:Entity {type: $type, value: $value})
                            MERGE (c)-[:MENTIONS]->(e)
                            """,
                            chunk_id=chunk_id,
                            type=entity_type,
                            value=entity_value,
                        )

            logger.debug(f"Indexed {indexed} entities for document {document_id}")
        except Exception as exc:
            logger.warning(f"Failed to index entities for {document_id}: {exc}")

        return indexed

    # ── Retrieval operations ─────────────────────────────────────────────────

    def expand_chunk_neighborhood(
        self,
        chunk_ids: List[str],
        hops: int = 1,
        document_id: Optional[str] = None,
    ) -> List[EvidenceChunk]:
        """
        Expand around a list of chunks by traversing the graph `hops` steps.
        Returns a list of neighboring chunks (with retrieval_method = 'graph').
        """
        if not self.enabled or not chunk_ids:
            return []

        expanded_chunks: List[EvidenceChunk] = []
        try:
            with self.driver.session() as session:
                # Query: find all chunks reachable from the input chunks
                # within `hops` relationship steps.
                cypher = f"""
                MATCH (c:Chunk WHERE c.id IN $chunk_ids)
                CALL apoc.path.subgraphAll(c, {{
                  relationshipFilter: "IN_DOCUMENT|MENTIONS|CO_OCCURS_WITH",
                  maxLevel: {hops}
                }}) YIELD nodes, relationships
                UNWIND nodes AS n
                WHERE n:Chunk AND n.id NOT IN $chunk_ids
                RETURN DISTINCT n.id AS chunk_id, n.document_id AS document_id,
                       n.text AS text, n.page AS page, n.section AS section
                """
                if document_id:
                    cypher += f" WHERE n.document_id = '{document_id}'"

                result = session.run(cypher, chunk_ids=chunk_ids)
                for record in result:
                    expanded_chunks.append(EvidenceChunk(
                        chunk_id=record["chunk_id"],
                        document_id=record["document_id"],
                        text=record["text"] or "",
                        page=record["page"],
                        section=record["section"],
                        score=0.5,  # Arbitrary score for graph-expanded chunks
                        retrieval_method="graph_expansion",
                    ))

            logger.debug(
                f"Expanded {len(chunk_ids)} chunks via graph → "
                f"found {len(expanded_chunks)} neighbors"
            )
        except Exception as exc:
            logger.warning(f"Graph neighborhood expansion failed: {exc}")

        return expanded_chunks

    def delete_document(self, document_id: str) -> bool:
        """Remove all nodes and relationships for a document."""
        if not self.enabled:
            return True

        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    DETACH DELETE d
                    """,
                    doc_id=document_id,
                )
            logger.debug(f"Deleted graph data for document {document_id}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to delete document {document_id} from graph: {exc}")
            return False

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


# ──────────────────────────────────────────────────────────────────────────────
# Singleton instance
# ──────────────────────────────────────────────────────────────────────────────

_GRAPH_STORE = None


def get_graph_store() -> GraphStore:
    """Get or create the graph store singleton."""
    global _GRAPH_STORE
    if _GRAPH_STORE is None:
        _GRAPH_STORE = GraphStore()
    return _GRAPH_STORE
