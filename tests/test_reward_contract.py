import unittest

import numpy as np

from reward import custom_reward, oasis_reward
from oasis_gym import OasisEnv


class RewardContractTest(unittest.TestCase):
    def test_botevade_reward_uses_named_transition_terms(self):
        self.assertEqual(
            custom_reward({"capture": 0.0, "goal_achieved": 0.0, "goal_distance": 0.4}),
            0.0,
        )
        self.assertEqual(
            custom_reward({"capture": 0.0, "goal_achieved": 0.0, "goal_distance": 0.01}),
            0.0,
        )
        self.assertEqual(
            custom_reward({"capture": 1.0, "goal_achieved": 0.0, "goal_distance": 0.4}),
            -1.0,
        )
        self.assertEqual(
            custom_reward({"capture": 0.0, "goal_achieved": 1.0, "goal_distance": 0.0}),
            1.0,
        )
        self.assertEqual(
            custom_reward({"puffed": 1.0, "goal_achieved": 0.0}),
            -1.0,
        )

    def test_flattened_observation_is_rejected(self):
        with self.assertRaises(TypeError):
            custom_reward(np.zeros(20, dtype=np.float32))

    def test_oasis_goal_bonus_is_an_event_not_a_position(self):
        reward = oasis_reward()
        no_event = reward(
            {
                "capture": 0.0,
                "goal_event": 0.0,
                "goal_distance": 0.5,
                "goals_remaining": 1.0,
                "finished": 0.0,
            }
        )
        goal_event = reward(
            {
                "capture": 0.0,
                "goal_event": 1.0,
                "goal_distance": 0.5,
                "goals_remaining": 1.0,
                "finished": 0.0,
            }
        )
        self.assertAlmostEqual(no_event, -0.025)
        self.assertAlmostEqual(goal_event, 0.975)

    def test_oasis_terminal_goal_transition_is_not_dropped(self):
        self.assertTrue(OasisEnv._goal_transitioned((0.05, 0.5), None))
        self.assertTrue(OasisEnv._goal_transitioned((0.05, 0.5), (0.2, 0.3)))
        self.assertFalse(OasisEnv._goal_transitioned((0.05, 0.5), (0.05, 0.5)))
        self.assertFalse(OasisEnv._goal_transitioned(None, (0.05, 0.5)))


if __name__ == "__main__":
    unittest.main()
