import contextlib
import io
import unittest
from unittest.mock import patch

import main
from nesyfs.config import Settings
from nesyfs.models import StepResult


class CommandLineTests(unittest.TestCase):
    base_arguments = ["main.py", "--session-id", "episode-1", "--task", "Put the apple in the fridge"]

    def assert_argument_error(self, arguments, message):
        stderr = io.StringIO()
        with patch("sys.argv", arguments), contextlib.redirect_stderr(stderr):
            with patch.object(main.Settings, "from_env") as settings:
                with self.assertRaises(SystemExit) as error:
                    main.main()
                self.assertEqual(error.exception.code, 2)
                settings.assert_not_called()
        self.assertIn(message, stderr.getvalue())

    def test_manual_mode_requires_nonblank_observation(self):
        for observation in (None, "  "):
            arguments = self.base_arguments.copy()
            if observation is not None:
                arguments += ["--observation", observation]
            with self.subTest(observation=observation):
                self.assert_argument_error(arguments, "--observation is required")

    def test_step_limit_must_be_positive(self):
        self.assert_argument_error(
            self.base_arguments + ["--environment", "kitchen-demo", "--max-steps", "0"],
            "--max-steps must be positive",
        )

    def test_blank_session_and_task_are_rejected_before_setup(self):
        for flag in ("--session-id", "--task"):
            arguments = self.base_arguments.copy()
            arguments[arguments.index(flag) + 1] = "  "
            with self.subTest(flag=flag):
                self.assert_argument_error(arguments, f"{flag} must be non-empty")

    def test_bad_environment_configuration_is_reported_without_traceback(self):
        stderr = io.StringIO()
        with patch("sys.argv", self.base_arguments + ["--environment", "kitchen-demo"]):
            with patch.object(main.Settings, "from_env", side_effect=ValueError("Missing OPENAI_API_KEY")):
                with patch.object(main, "Neo4jKnowledgeGraph") as graph, contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        main.main()
                    self.assertEqual(error.exception.code, 2)
                    graph.assert_not_called()
        self.assertIn("Missing OPENAI_API_KEY", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_help_does_not_read_credentials(self):
        with patch("sys.argv", ["main.py", "--help"]), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(main.Settings, "from_env") as settings:
                with self.assertRaises(SystemExit) as error:
                    main.main()
                self.assertEqual(error.exception.code, 0)
                settings.assert_not_called()

    def test_graph_is_closed_if_connection_check_fails(self):
        settings = Settings("test-key", "test-model", "bolt://example.test:7687", "test-user", "test-password")
        with patch("sys.argv", self.base_arguments + ["--environment", "kitchen-demo"]):
            with patch.object(main.Settings, "from_env", return_value=settings):
                with patch.object(main, "Neo4jKnowledgeGraph") as graph:
                    graph.return_value.verify_connection.side_effect = RuntimeError("offline")
                    with self.assertRaisesRegex(RuntimeError, "offline"):
                        main.main()
                    graph.return_value.close.assert_called_once_with()

    def test_manual_mode_performs_one_step(self):
        settings = Settings("test-key", "test-model", "bolt://example.test:7687", "test-user", "test-password")
        arguments = self.base_arguments + ["--observation", "An apple is on the table.", "--max-steps", "5"]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(main.Settings, "from_env", return_value=settings):
                with patch.object(main, "Neo4jKnowledgeGraph") as graph:
                    with patch.object(main, "NeSyFSLanguageModel"), patch.object(main, "TSMCPlanner"):
                        with patch.object(main, "NeSyFSAgent") as agent:
                            agent.return_value.step.return_value = StepResult(
                                action="take apple", observation="You hold the apple.",
                                planning_used=False, reflection_attempts=1,
                            )
                            main.main()
                            agent.return_value.step.assert_called_once()
                    graph.return_value.close.assert_called_once_with()
