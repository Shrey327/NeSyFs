import unittest
from unittest.mock import MagicMock, Mock, patch

from nesyfs.knowledge_graph import Neo4jKnowledgeGraph
from nesyfs.models import ExtractedFact, MemoryDelta


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.driver = MagicMock()
        self.session = self.driver.session.return_value.__enter__.return_value
        self.driver_patch = patch("nesyfs.knowledge_graph.GraphDatabase.driver", return_value=self.driver)
        self.factory = self.driver_patch.start()
        self.addCleanup(self.driver_patch.stop)
        self.graph = Neo4jKnowledgeGraph("bolt://example.test:7687", "test-user", "test-password", 10)

    def test_driver_lifecycle(self):
        self.factory.assert_called_once_with("bolt://example.test:7687", auth=("test-user", "test-password"))
        self.graph.verify_connection()
        self.graph.close()
        self.driver.verify_connectivity.assert_called_once_with()
        self.driver.close.assert_called_once_with()

    def test_setup_creates_session_scoped_constraints(self):
        self.graph.setup()
        queries = [call.args[0] for call in self.session.run.call_args_list]
        self.assertEqual(len(queries), 2)
        self.assertTrue(all("IF NOT EXISTS" in query for query in queries))
        self.assertIn("(e.session_id, e.name) IS UNIQUE", queries[0])
        self.assertIn("(f.session_id, f.id) IS UNIQUE", queries[1])
        self.assertEqual(self.session.run.return_value.consume.call_count, 2)

    def test_empty_entity_list_skips_query(self):
        self.assertEqual(self.graph.retrieve("episode-1", []), [])
        self.driver.session.assert_not_called()

    def test_retrieve_uses_parameters_for_session_and_names(self):
        session_id = "episode-'quoted'"
        entity_name = "apple' MATCH (n)"
        record = Mock()
        record.data.return_value = {
            "id": "fact-1", "subject": "apple", "predicate": "on", "object": "table",
            "observed_at": "now", "evidence": "An apple is on the table.",
        }
        self.session.run.return_value = [record]
        facts = self.graph.retrieve(session_id, [entity_name, entity_name])
        query = self.session.run.call_args.args[0]
        parameters = self.session.run.call_args.kwargs
        self.assertNotIn(session_id, query)
        self.assertNotIn(entity_name, query)
        self.assertIn("NeSyEntity {session_id: $session_id}", query)
        self.assertIn("NeSyFact {session_id: $session_id}", query)
        self.assertEqual(parameters, {"session_id": session_id, "names": [entity_name], "limit": 10})
        self.assertEqual(facts[0].id, "fact-1")

    def test_upsert_submits_atomic_delta_with_bounded_evidence(self):
        delta = MemoryDelta(
            facts=[ExtractedFact(subject="apple", predicate="on", object="table")],
            superseded_fact_ids=["stale-1"],
        )
        self.graph.upsert("episode-1", delta, "x" * 600)
        callback, session_id, stale_ids, facts = self.session.execute_write.call_args.args
        self.assertEqual(callback, self.graph._apply_delta)
        self.assertEqual(session_id, "episode-1")
        self.assertEqual(stale_ids, ["stale-1"])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["evidence"], "x" * 500)
        self.assertEqual(facts[0]["subject"], "apple")
        self.assertTrue(facts[0]["id"])
        self.assertTrue(facts[0]["observed_at"])

    def test_delta_removal_and_insertion_are_session_scoped(self):
        tx = MagicMock()
        facts = [{"id": "new-1", "subject": "apple", "predicate": "on", "object": "table"}]
        self.graph._apply_delta(tx, "episode-1", ["stale-1"], facts)
        delete, insert = tx.run.call_args_list
        self.assertIn("NeSyFact {session_id: $session_id}", delete.args[0])
        self.assertEqual(delete.kwargs, {"session_id": "episode-1", "stale_ids": ["stale-1"]})
        self.assertEqual(insert.args[0].count("session_id: $session_id"), 3)
        self.assertEqual(insert.kwargs, {"session_id": "episode-1", "facts": facts})
        self.assertEqual(tx.run.return_value.consume.call_count, 2)
