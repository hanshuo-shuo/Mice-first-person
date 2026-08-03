import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from first_person import FirstPersonRenderer, FirstPersonVisionWrapper


class _Polygon:
    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float64)


class _State:
    def __init__(self, location, direction=0.0):
        self.location = location
        self.direction = direction
        self.velocity = (0.0, 0.0)


class _Dynamics:
    v_max = 0.5
    accel_scale = 6.0
    damping = 8.0


class _Agent:
    def __init__(self, location, direction=0.0):
        self.state = _State(location, direction)
        self.dynamics = _Dynamics()


class _Model:
    def __init__(self):
        self.arena = _Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.occlusions = [_Polygon([(0.48, 0.35), (0.58, 0.35), (0.58, 0.65), (0.48, 0.65)])]
        self.prey = _Agent((0.20, 0.50), 0.0)
        self.predator = _Agent((0.75, 0.50), 180.0)
        self.use_predator = True
        self.goal_location = (0.92, 0.72)
        self.running = True

    def stop(self):
        self.running = False


class _DummyEnv(gym.Env):
    def __init__(self):
        self.model = _Model()
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        self.time_step = 0.1
        self.steps = 0
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.model.prey.state.location = (0.20, 0.50)
        self.model.prey.state.direction = 0.0
        self.model.prey.state.velocity = (0.0, 0.0)
        return np.asarray(self.model.prey.state.location, dtype=np.float32), {"kept": True}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        self.steps += 1
        x, y = self.model.prey.state.location
        self.model.prey.state.location = (x + float(action[0]) * 0.01, y)
        self.model.prey.state.velocity = (float(action[0]), float(action[1]))
        state = np.asarray(self.model.prey.state.location, dtype=np.float32)
        return state, 1.25, False, self.steps >= 3, {"step": self.steps}


class FirstPersonRendererTest(unittest.TestCase):
    def test_rgb_frame_contract_and_camera_rotation(self):
        model = _Model()
        renderer = FirstPersonRenderer(model, width=96, height=64, horizontal_fov=90)

        east_frame = renderer.render_rgb()
        self.assertEqual(east_frame.shape, (64, 96, 3))
        self.assertEqual(east_frame.dtype, np.uint8)
        self.assertGreater(float(east_frame.std()), 10.0)

        model.prey.state.direction = 90.0
        north_frame = renderer.render_rgb()
        self.assertFalse(np.array_equal(east_frame, north_frame))

    def test_positive_world_y_is_camera_left_when_heading_east(self):
        model = _Model()
        model.occlusions = []
        model.use_predator = False
        model.goal_location = (0.70, 0.70)
        renderer = FirstPersonRenderer(model, width=120, height=80, horizontal_fov=90)

        frame = renderer.render_rgb()
        green = (frame[..., 1] > 170) & (frame[..., 1] > frame[..., 0] * 1.5)
        _, green_x = np.nonzero(green)
        self.assertGreater(len(green_x), 0)
        self.assertLess(float(green_x.mean()), frame.shape[1] / 2.0)

    def test_wrapper_only_replaces_observation(self):
        env = FirstPersonVisionWrapper(
            _DummyEnv(),
            width=80,
            height=60,
            render_mode="rgb_array",
            observation_mode="single_rgb",
            action_mode="passthrough",
        )
        obs, info = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info, {"kept": True})
        np.testing.assert_allclose(env.state_observation, [0.20, 0.50])

        obs, reward, terminated, truncated, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(reward, 1.25)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {"step": 1})
        np.testing.assert_allclose(env.state_observation, [0.21, 0.50])
        self.assertEqual(env.render().shape, (60, 80, 3))
        env.close()

    def test_mouse_observation_and_egocentric_action_contract(self):
        env = FirstPersonVisionWrapper(
            _DummyEnv(),
            width=80,
            height=60,
            render_mode="rgb_array",
            observation_mode="mouse",
            action_mode="egocentric_velocity",
        )
        obs, info = env.reset(seed=3)
        self.assertEqual(info, {"kept": True})
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs["image_left"].shape, (60, 80, 3))
        self.assertEqual(obs["image_right"].shape, (60, 80, 3))
        self.assertFalse(np.array_equal(obs["image_left"], obs["image_right"]))
        np.testing.assert_allclose(obs["previous_action"], [0.0, 0.0])

        action = np.asarray([1.0, 0.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertGreater(float(env.unwrapped.last_action[0]), 0.0)
        self.assertAlmostEqual(float(env.unwrapped.last_action[1]), 0.0)
        np.testing.assert_allclose(obs["previous_action"], action)
        self.assertEqual(env.render().shape, (60, 164, 3))
        env.close()

    def test_body_and_head_can_turn_without_translation(self):
        env = FirstPersonVisionWrapper(
            _DummyEnv(),
            width=80,
            height=60,
            observation_mode="mouse",
            action_mode="egocentric_velocity_head",
        )
        before, _ = env.reset()
        start_location = tuple(env.unwrapped.model.prey.state.location)
        after, _, _, _, _ = env.step(
            np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(tuple(env.unwrapped.model.prey.state.location), start_location)
        self.assertAlmostEqual(env.body_heading_degrees, 18.0)
        self.assertAlmostEqual(env.head_yaw_degrees, 24.0)
        self.assertFalse(np.array_equal(before["image_left"], after["image_left"]))
        np.testing.assert_allclose(after["previous_action"], [0.0, 1.0, 1.0])
        env.close()


if __name__ == "__main__":
    unittest.main()
