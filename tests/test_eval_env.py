import unittest

from nesyfs.eval_env import EvaluationKitchen, SCENARIOS


class EvaluationKitchenTests(unittest.TestCase):
    def test_each_scenario_reaches_goal_through_real_state_changes(self):
        actions = {
            "apple_table_fridge": ["take apple", "open fridge", "put apple in fridge"],
            "pear_cupboard_fridge": ["open cupboard", "take pear", "put pear in fridge"],
            "mug_counter_cupboard": ["take mug", "put mug in cupboard"],
            "apple_fridge_cupboard": ["open fridge", "take apple", "open cupboard", "put apple in cupboard"],
            "key_unlock_cupboard": ["take key", "unlock cupboard with key", "put key on table", "open cupboard", "take apple", "put apple in cupboard", "close cupboard"],
            "wash_pear_fridge": ["take pear", "turn on tap", "wash pear", "open fridge", "put pear in fridge", "turn off tap"],
            "two_items_cupboard": ["open fridge", "take apple", "open cupboard", "put apple in cupboard", "close fridge", "take pear", "put pear in cupboard"],
            "wash_mug_secure": ["take key", "unlock cupboard with key", "put key on counter", "open cupboard", "take mug", "turn on tap", "wash mug", "put mug in fridge", "close cupboard", "turn off tap"],
        }
        self.assertEqual(set(actions), {s.id for s in SCENARIOS})
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.id):
                env = EvaluationKitchen(scenario)
                self.assertFalse(env.is_complete)
                self.assertEqual(len(actions[scenario.id]), scenario.min_actions)
                for action in actions[scenario.id]:
                    self.assertFalse(env.is_complete)
                    env.execute(action)
                self.assertTrue(env.is_complete)
                self.assertEqual(env.invalid_actions, 0)

    def test_failed_and_repeated_actions_count_without_false_success(self):
        env = EvaluationKitchen(SCENARIOS[1])
        self.assertIn("closed", env.execute("take pear"))
        env.execute("open cupboard")
        self.assertIn("already open", env.execute("open cupboard"))
        env.execute("take pear")
        self.assertIn("not holding", env.execute("put apple in fridge"))
        self.assertIn("Invalid", env.execute("teleport pear"))
        self.assertEqual(env.invalid_actions, 4)
        self.assertFalse(env.is_complete)

    def test_outcomes_are_partial_and_instances_are_isolated(self):
        first, second = EvaluationKitchen(SCENARIOS[0]), EvaluationKitchen(SCENARIOS[0])
        initial = first.observe()
        self.assertEqual(initial, second.observe())
        self.assertIn("pear", initial)
        outcome = first.execute("open the fridge.")
        self.assertEqual(first.observe(), outcome)
        self.assertNotIn("pear", outcome)
        self.assertEqual(second.observe(), initial)
        self.assertEqual(second.invalid_actions, 0)

    def test_locked_container_requires_held_key_and_closing_is_a_goal(self):
        scenario = next(s for s in SCENARIOS if s.id == "key_unlock_cupboard")
        env = EvaluationKitchen(scenario)
        self.assertIn("locked", env.observe())
        self.assertIn("locked", env.execute("open cupboard"))
        self.assertIn("hold the key", env.execute("unlock cupboard with key"))
        for action in ["take key", "unlock cupboard with key", "put key on table", "open cupboard", "take apple", "put apple in cupboard"]:
            env.execute(action)
        self.assertFalse(env.is_complete)
        env.execute("close cupboard")
        self.assertTrue(env.is_complete)
        self.assertEqual(env.invalid_actions, 2)

    def test_clean_and_tap_goals_use_state_not_the_last_message(self):
        scenario = next(s for s in SCENARIOS if s.id == "wash_pear_fridge")
        env = EvaluationKitchen(scenario)
        env.execute("take pear")
        self.assertIn("tap is off", env.execute("wash pear"))
        env.execute("turn on tap")
        env.execute("wash pear")
        self.assertIn("already clean", env.execute("wash pear"))
        env.execute("open fridge")
        env.execute("put pear in fridge")
        self.assertFalse(env.is_complete)
        env.execute("turn off tap")
        self.assertTrue(env.is_complete)
        self.assertEqual(env.invalid_actions, 2)

    def test_two_object_goal_and_wrong_source_are_checked(self):
        scenario = next(s for s in SCENARIOS if s.id == "two_items_cupboard")
        env = EvaluationKitchen(scenario)
        self.assertIn("not in or on", env.execute("take pear from counter"))
        for action in ["open fridge", "take apple", "open cupboard", "put apple in cupboard", "close fridge"]:
            env.execute(action)
        self.assertFalse(env.is_complete)
        env.execute("take pear")
        env.execute("put pear in cupboard")
        self.assertTrue(env.is_complete)
        self.assertEqual(env.invalid_actions, 1)

    def test_unknown_or_redundant_controls_are_invalid(self):
        env = EvaluationKitchen(next(s for s in SCENARIOS if s.id == "wash_mug_secure"))
        for action in ["close cupboard", "turn off tap", "unlock fridge with key", "put key on table", "dance"]:
            env.execute(action)
        self.assertEqual(env.invalid_actions, 5)
        self.assertFalse(env.is_complete)


if __name__ == "__main__":
    unittest.main()
