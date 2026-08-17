import sys
import unittest
from pathlib import Path


CELLWORLD_PATH = str(Path(__file__).resolve().parents[1] / "cellworld_game-main")
if CELLWORLD_PATH not in sys.path:
    sys.path.insert(0, CELLWORLD_PATH)

try:
    import cellworld_game as cwgame
except ModuleNotFoundError as import_error:  # pragma: no cover - optional runtime dependency
    cwgame = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"Cellworld runtime unavailable: {_IMPORT_ERROR}")
class BodyHeadingModelTest(unittest.TestCase):
    def make_model(self):
        model = cwgame.Model(
            world_name="unit",
            arena=cwgame.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            occlusions=[],
            time_step=0.1,
        )
        mouse = cwgame.Mouse(
            start_state=cwgame.AgentState(
                location=(0.5, 0.5),
                body_heading=90.0,
            ),
            max_forward_speed=0.5,
            accel_scale=1.0,
            damping=0.0,
        )
        model.add_agent("prey", mouse)
        model.reset()
        return model, mouse

    def test_point_mass_velocity_does_not_rotate_body(self):
        model, mouse = self.make_model()
        try:
            mouse.set_action(1.0, 0.0)
            model.step()

            self.assertAlmostEqual(mouse.state.body_heading, 90.0)
            self.assertAlmostEqual(mouse.state.direction, 90.0)
            self.assertGreater(mouse.state.velocity[0], 0.0)
            self.assertAlmostEqual(mouse.state.velocity[1], 0.0)
        finally:
            model.close()

    def test_model_snapshot_uses_canonical_heading_field(self):
        model, mouse = self.make_model()
        try:
            snapshot = model.get_state_dict()
            state = snapshot["agents"]["prey"]["state"]
            self.assertEqual(state["body_heading"], 90.0)
            self.assertNotIn("direction", state)

            mouse.state.direction = 15.0
            self.assertEqual(mouse.state.body_heading, 15.0)
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main()
