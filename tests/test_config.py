import os
import unittest
from unittest.mock import patch

from nesyfs.config import Settings


class SettingsTests(unittest.TestCase):
    credentials = {"OPENAI_API_KEY": "test-key", "NEO4J_PASSWORD": "test-password"}

    def test_defaults(self):
        with patch.dict(os.environ, self.credentials, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.model, "gpt-4.1-mini")
        self.assertEqual(settings.neo4j_uri, "bolt://localhost:7687")
        self.assertEqual(settings.neo4j_username, "neo4j")
        self.assertEqual(settings.reflection_limit, 3)
        self.assertEqual(settings.reflection_samples, 5)
        self.assertEqual(settings.particle_count, 6)
        self.assertEqual(settings.planning_horizon, 5)
        self.assertEqual(settings.retrieval_limit, 100)

    def test_environment_overrides(self):
        values = {
            **self.credentials,
            "NESYFS_MODEL": "test-model",
            "NEO4J_URI": "bolt://example.test:7687",
            "NEO4J_USERNAME": "test-user",
            "NESYFS_REFLECTION_LIMIT": "2",
            "NESYFS_REFLECTION_SAMPLES": "3",
            "NESYFS_PARTICLE_COUNT": "4",
            "NESYFS_PLANNING_HORIZON": "2",
            "NESYFS_RETRIEVAL_LIMIT": "10",
        }
        with patch.dict(os.environ, values, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.model, "test-model")
        self.assertEqual(settings.neo4j_uri, "bolt://example.test:7687")
        self.assertEqual(settings.neo4j_username, "test-user")
        self.assertEqual(settings.reflection_limit, 2)
        self.assertEqual(settings.reflection_samples, 3)
        self.assertEqual(settings.particle_count, 4)
        self.assertEqual(settings.planning_horizon, 2)
        self.assertEqual(settings.retrieval_limit, 10)

    def test_missing_credentials_are_named(self):
        for name in self.credentials:
            values = {key: value for key, value in self.credentials.items() if key != name}
            with self.subTest(name=name), patch.dict(os.environ, values, clear=True):
                with self.assertRaisesRegex(ValueError, name):
                    Settings.from_env()

    def test_empty_credentials_are_rejected(self):
        for name in self.credentials:
            values = {**self.credentials, name: ""}
            with self.subTest(name=name), patch.dict(os.environ, values, clear=True):
                with self.assertRaisesRegex(ValueError, name):
                    Settings.from_env()

    def test_nonpositive_limits_are_rejected(self):
        names = (
            "NESYFS_REFLECTION_LIMIT",
            "NESYFS_REFLECTION_SAMPLES",
            "NESYFS_PARTICLE_COUNT",
            "NESYFS_PLANNING_HORIZON",
            "NESYFS_RETRIEVAL_LIMIT",
        )
        for name in names:
            for value in ("0", "-1"):
                with self.subTest(name=name, value=value):
                    with patch.dict(os.environ, {**self.credentials, name: value}, clear=True):
                        with self.assertRaisesRegex(ValueError, "positive integers"):
                            Settings.from_env()

    def test_noninteger_limit_is_rejected(self):
        with patch.dict(os.environ, {**self.credentials, "NESYFS_PARTICLE_COUNT": "six"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()
