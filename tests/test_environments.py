import unittest

from nesyfs import CallableEnvironment, KitchenDemoEnvironment


class KitchenDemoTests(unittest.TestCase):
    def setUp(self):
        self.environment = KitchenDemoEnvironment()

    def test_initial_state(self):
        self.assertFalse(self.environment.is_complete)
        self.assertIn("apple is on the table", self.environment.observe())
        self.assertIn("fridge is closed", self.environment.observe())

    def test_open_fridge_before_taking_apple_is_observed(self):
        self.environment.execute("open fridge")
        self.assertIn("fridge is open", self.environment.observe())
        self.assertIn("fridge is open", self.environment.execute("look around"))

    def test_complete_episode(self):
        self.environment.execute("take apple")
        self.environment.execute("open fridge")
        self.environment.execute("put apple in fridge")
        self.assertTrue(self.environment.is_complete)
        self.assertFalse(self.environment.holding_apple)
        self.assertEqual(self.environment.observe(), "The apple is inside the fridge.")

    def test_cannot_store_apple_without_holding_it(self):
        self.environment.execute("open fridge")
        observation = self.environment.execute("put apple in fridge")
        self.assertEqual(observation, "You are not holding the apple.")
        self.assertFalse(self.environment.is_complete)
        self.assertEqual(self.environment.apple_location, "table")

    def test_closed_fridge_rejects_placement_without_losing_apple(self):
        self.environment.execute("take apple")
        observation = self.environment.execute("put apple in fridge")
        self.assertIn("fridge is closed", observation)
        self.assertTrue(self.environment.holding_apple)
        self.assertFalse(self.environment.is_complete)

    def test_unknown_action_does_not_change_state(self):
        before = self.environment.observe()
        self.assertIn("action had no effect", self.environment.execute("dance"))
        self.assertEqual(self.environment.observe(), before)

    def test_case_insensitive_action_aliases(self):
        self.environment.execute("GRAB APPLE")
        self.environment.execute("OPEN REFRIGERATOR")
        self.environment.execute("STORE APPLE IN REFRIGERATOR")
        self.assertTrue(self.environment.is_complete)


class CallableEnvironmentTests(unittest.TestCase):
    def test_passes_action_to_callback(self):
        actions = []

        def execute(action):
            actions.append(action)
            return "The door is open."

        environment = CallableEnvironment(execute)
        self.assertEqual(environment.execute("open door"), "The door is open.")
        self.assertEqual(actions, ["open door"])

    def test_rejects_invalid_observations(self):
        for result in (None, 42, "", "  "):
            with self.subTest(result=result):
                environment = CallableEnvironment(lambda action: result)
                with self.assertRaisesRegex(ValueError, "non-empty observation"):
                    environment.execute("open door")
