from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from neo4j import Driver, GraphDatabase

from nesyfs.models import ExtractedFact, Fact, MemoryDelta


class Neo4jKnowledgeGraph:
    """Session-scoped fact graph; all query values are parameterized."""

    def __init__(self, uri: str, username: str, password: str, retrieval_limit: int = 100):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
        self._retrieval_limit = retrieval_limit

    def close(self) -> None:
        self._driver.close()

    def verify_connection(self) -> None:
        self._driver.verify_connectivity()

    def setup(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT nesyfs_entity_key IF NOT EXISTS "
                "FOR (e:NeSyEntity) REQUIRE (e.session_id, e.name) IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT nesyfs_fact_key IF NOT EXISTS "
                "FOR (f:NeSyFact) REQUIRE (f.session_id, f.id) IS UNIQUE"
            ).consume()

    def retrieve(self, session_id: str, entity_names: list[str]) -> list[Fact]:
        if not entity_names:
            return []
        with self._driver.session() as session:
            records = session.run(
                "MATCH (e:NeSyEntity {session_id: $session_id}) "
                "WHERE e.name IN $names "
                "WITH collect(e) AS seeds "
                "UNWIND seeds AS e "
                "MATCH (e)-[:HAS_FACT]->(f:NeSyFact {session_id: $session_id}) "
                "RETURN DISTINCT f.id AS id, f.subject AS subject, "
                "f.predicate AS predicate, f.object AS object, f.observed_at AS observed_at, "
                "f.evidence AS evidence "
                "ORDER BY f.observed_at DESC LIMIT $limit",
                session_id=session_id,
                names=list(dict.fromkeys(entity_names)),
                limit=self._retrieval_limit,
            )
            return [Fact.model_validate(record.data()) for record in records]

    def upsert(self, session_id: str, delta: MemoryDelta, observation: str) -> None:
        facts = [
            {
                "id": str(uuid4()),
                **fact.model_dump(),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "evidence": observation[:500],
            }
            for fact in delta.facts
        ]
        with self._driver.session() as session:
            session.execute_write(self._apply_delta, session_id, delta.superseded_fact_ids, facts)

    @staticmethod
    def _apply_delta(tx, session_id: str, stale_ids: list[str], facts: list[dict]) -> None:
        tx.run(
            "MATCH (f:NeSyFact {session_id: $session_id}) "
            "WHERE f.id IN $stale_ids DETACH DELETE f",
            session_id=session_id,
            stale_ids=stale_ids,
        ).consume()
        tx.run(
            "UNWIND $facts AS row "
            "MERGE (s:NeSyEntity {session_id: $session_id, name: row.subject}) "
            "MERGE (f:NeSyFact {session_id: $session_id, id: row.id}) "
            "SET f.subject = row.subject, f.predicate = row.predicate, "
            "f.object = row.object, f.observed_at = row.observed_at, f.evidence = row.evidence "
            "MERGE (s)-[:HAS_FACT]->(f) "
            "MERGE (o:NeSyEntity {session_id: $session_id, name: row.object}) "
            "MERGE (o)-[:HAS_FACT]->(f)",
            session_id=session_id,
            facts=facts,
        ).consume()
