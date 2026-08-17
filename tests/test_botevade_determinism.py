import copy
import unittest
from pathlib import Path

import numpy as np


try:
    import cellworld.util as cellworld_util
    from botevade_gym import BotEvadeEnv, FirstPersonBotEvadeEnv
except ModuleNotFoundError as import_error:  # pragma: no cover - optional runtime dependency
    cellworld_util = None
    BotEvadeEnv = None
    FirstPersonBotEvadeEnv = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"Cellworld runtime unavailable: {_IMPORT_ERROR}")
class BotEvadeDeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_cache_folder = cellworld_util.cellworld_cache_folder
        cellworld_util.cellworld_cache_folder = str(
            Path(__file__).resolve().parents[1] / "cellworld_cache"
        )

    @classmethod
    def tearDownClass(cls):
        cellworld_util.cellworld_cache_folder = cls._old_cache_folder

    @staticmethod
    def make_env():
        return BotEvadeEnv(
            world_name="21_05",
            use_lppos=False,
            use_predator=True,
            action_type=BotEvadeEnv.ActionType.CONTINUOUS,
            time_step=0.25,
            max_step=40,
            render=False,
        )

    @staticmethod
    def physical_trace(env):
        model = env.model
        predator = model.predator
        return (
            tuple(model.prey.state.location),
            tuple(model.prey.state.velocity),
            tuple(model.predator.state.location),
            float(model.predator.state.body_heading),
            tuple(predator.path),
            predator.new_destination,
            predator.destination,
            int(predator.destination_wait),
            int(predator.navigation_plan_update_wait),
            float(predator.dynamics.forward_speed),
            float(predator.dynamics.turn_speed),
            float(model.time),
            int(model.step_count),
        )

    def rollout(self, seed):
        env = self.make_env()
        try:
            observation, _ = env.reset(seed=seed)
            records = [(np.array(observation, copy=True), self.physical_trace(env))]
            for action in (
                np.asarray((0.1, -0.2), dtype=np.float32),
                np.asarray((0.5, 0.0), dtype=np.float32),
                np.asarray((-0.3, 0.4), dtype=np.float32),
            ):
                observation, reward, terminated, truncated, info = env.step(action)
                records.append(
                    (
                        np.array(observation, copy=True),
                        float(reward),
                        bool(terminated),
                        bool(truncated),
                        copy.deepcopy(info),
                        self.physical_trace(env),
                    )
                )
            return records
        finally:
            env.close()

    def test_reset_seed_replays_identical_predator_trajectory(self):
        first = self.rollout(seed=23)
        second = self.rollout(seed=23)
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])

    def test_reset_reinitializes_time_and_navigation_state(self):
        env = self.make_env()
        try:
            env.reset(seed=23)
            env.step(np.asarray((1.0, 0.0), dtype=np.float32))
            self.assertGreater(env.model.time, 0.0)
            self.assertGreater(env.model.step_count, 0)

            env.model.predator.destination = (0.01, 0.01)
            env.model.predator.path = [(0.01, 0.01)]
            env.model.predator.new_destination = (0.02, 0.02)
            env.model.predator.destination_wait = 9
            env.model.predator.navigation_plan_update_wait = 8
            env.model.predator.dynamics.forward_speed = 7
            env.model.predator.dynamics.turn_speed = 6

            env.reset(seed=23)
            predator = env.model.predator
            self.assertEqual(env.model.time, 0)
            self.assertEqual(env.model.step_count, 0)
            self.assertIsNone(predator.destination)
            self.assertEqual(predator.path, [])
            self.assertNotEqual(predator.new_destination, (0.02, 0.02))
            self.assertEqual(predator.destination_wait, 0)
            self.assertEqual(predator.navigation_plan_update_wait, 0)
            self.assertEqual(predator.dynamics.forward_speed, 0)
            self.assertEqual(predator.dynamics.turn_speed, 0)
        finally:
            env.close()

    def test_state_dict_round_trip_replays_counterfactual(self):
        env = self.make_env()
        try:
            env.reset(seed=23)
            env.step(np.asarray((0.1, -0.2), dtype=np.float32))
            env.step(np.asarray((0.5, 0.0), dtype=np.float32))
            snapshot = env.get_state_dict()

            predator_state = snapshot["model"]["agents"]["predator"]
            self.assertIn("state", predator_state)
            self.assertIn("dynamics", predator_state)
            self.assertIn("path", predator_state)
            self.assertIn("destination_wait", predator_state)
            self.assertIn("navigation_plan_update_wait", predator_state)
            self.assertIn("python", snapshot["rng"])
            self.assertIn("numpy", snapshot["rng"])
            self.assertIn("gymnasium", snapshot["rng"])
            self.assertIn("puff_cool_down", snapshot["model"]["task"])

            action = np.asarray((-0.3, 0.4), dtype=np.float32)
            first = env.step(action)
            first_trace = (np.array(first[0], copy=True), first[1:], self.physical_trace(env))

            env.set_state_dict(snapshot)
            second = env.step(action)
            second_trace = (np.array(second[0], copy=True), second[1:], self.physical_trace(env))

            np.testing.assert_array_equal(first_trace[0], second_trace[0])
            self.assertEqual(first_trace[1:], second_trace[1:])
        finally:
            env.close()

    def test_first_person_snapshot_includes_embodiment_state(self):
        env = FirstPersonBotEvadeEnv(
            world_name="21_05",
            use_lppos=False,
            use_predator=True,
            action_mode="egocentric_velocity_head",
            vision_width=32,
            vision_height=32,
            render_mode="rgb_array",
        )
        try:
            env.reset(seed=23)
            env.step(np.asarray((0.4, 0.2, -0.5), dtype=np.float32))
            snapshot = env.get_state_dict()
            self.assertIn("first_person", snapshot)
            self.assertIn("body_heading_degrees", snapshot["first_person"])
            self.assertIn("head_yaw_degrees", snapshot["first_person"])
            self.assertIn("previous_action", snapshot["first_person"])

            action = np.asarray((-0.2, -0.3, 0.6), dtype=np.float32)
            first = env.step(action)
            env.set_state_dict(snapshot)
            second = env.step(action)
            for left, right in zip(first[0].values(), second[0].values()):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(first[1:], second[1:])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
