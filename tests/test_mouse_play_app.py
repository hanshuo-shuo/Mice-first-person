import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mouse_play_app import DemonstrationRecorder, _resource_specs


class DemonstrationRecorderTest(unittest.TestCase):
    @staticmethod
    def observation(value: int):
        return {
            "image_left": np.full((12, 16, 3), value, dtype=np.uint8),
            "image_right": np.full((12, 16, 3), value + 1, dtype=np.uint8),
            "proprio": np.asarray((0.1, -0.2, 0.3), dtype=np.float32),
            "previous_action": np.zeros(3, dtype=np.float32),
        }

    def test_records_pickle_free_transition_file_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = DemonstrationRecorder(
                data_root=Path(temporary),
                session_metadata={
                    "action_names": [
                        "forward_velocity",
                        "body_yaw_rate",
                        "head_yaw_rate",
                    ],
                },
                max_steps_per_file=10,
            )
            session = recorder.start()
            self.assertTrue(session.is_dir())
            action = np.asarray((0.5, -0.25, 0.0), dtype=np.float32)
            recorder.record_transition(
                observation=self.observation(10),
                action=action,
                reward=1.0,
                terminated=False,
                truncated=False,
                sim_time=0.0,
                privileged_state=np.arange(9, dtype=np.float32),
            )
            saved = recorder.record_transition(
                observation=self.observation(20),
                action=action,
                reward=2.0,
                terminated=True,
                truncated=False,
                sim_time=0.1,
                privileged_state=np.arange(9, dtype=np.float32) + 1,
                final_info={"is_success": True, "captures": 0},
            )
            self.assertIsNotNone(saved)
            with np.load(saved, allow_pickle=False) as episode:
                self.assertEqual(episode["image_left"].shape, (2, 12, 16, 3))
                self.assertEqual(episode["action"].shape, (2, 3))
                self.assertEqual(episode["privileged_state"].shape, (2, 9))
                self.assertAlmostEqual(float(episode["reward"].sum()), 3.0)
                self.assertTrue(bool(episode["terminated"][-1]))

            episode_metadata = json.loads(saved.with_suffix(".json").read_text())
            self.assertEqual(episode_metadata["steps"], 2)
            self.assertTrue(episode_metadata["is_success"])
            session_metadata = json.loads((session / "session.json").read_text())
            self.assertEqual(session_metadata["episodes"][0]["file"], saved.name)
            self.assertIsNone(recorder.stop(reason="test_complete"))

    def test_offline_resource_manifest_contains_required_map_parts(self):
        resources = {path: optional for path, optional in _resource_specs("21_05")}
        self.assertFalse(resources["world_configuration/hexagonal"])
        self.assertFalse(resources["paths/hexagonal.21_05.astar.robot"])
        self.assertFalse(resources["graph/hexagonal.21_05.cell_visibility"])
        self.assertTrue(resources["cell_group/hexagonal.21_05.lppo"])


if __name__ == "__main__":
    unittest.main()
