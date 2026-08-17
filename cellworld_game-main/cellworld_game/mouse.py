import numpy as np
import pygame
from .agent import Agent, AgentState, PointDynamics
from .resources import Resources
from .polygon import Polygon


class Mouse(Agent):
    """Prey as a 2D point mass (PointMaze-style).

    The action interface is `set_action(ax, ay)` where `(ax, ay) ∈ [-1, 1]²`
    is interpreted as a desired acceleration. Integration, damping, and
    slide-on-collision all happen inside `Model._point_step`.

    The `navigation` argument is accepted and ignored — it is kept in the
    signature only so existing `Mouse(..., navigation=...)` call sites
    (e.g. `tasks/botevade.py`, `tasks/dualevade.py`) keep working.
    """

    def __init__(self,
                 start_state: AgentState,
                 navigation=None,
                 view_field: float = 360,
                 max_forward_speed: float = 0.5,
                 max_turning_speed: float = 20.0,  # accepted, unused
                 accel_scale: float = 6.0,
                 damping: float = 8.0):
        Agent.__init__(self,
                       view_field=view_field,
                       size=0.04,
                       sprite_scale=2.0,
                       polygon_color=(20, 90, 20))
        self.start_state = start_state
        # keep `max_forward_speed` as the public name so downstream code
        # reading `prey.max_forward_speed` (e.g. BotEvade) still works
        self.max_forward_speed = max_forward_speed
        self.max_turning_speed = max_turning_speed
        self.dynamics = PointDynamics(accel_scale=accel_scale,
                                      v_max=max_forward_speed,
                                      damping=damping)
        self.collision = True
        self.visible = True

    def set_action(self, ax: float, ay: float) -> None:
        self.dynamics.ax = float(np.clip(ax, -1.0, 1.0))
        self.dynamics.ay = float(np.clip(ay, -1.0, 1.0))

    def stop(self) -> None:
        self.dynamics.ax = 0.0
        self.dynamics.ay = 0.0

    # --- back-compat shim: some old demos call set_destination ---------- #
    def set_destination(self, destination) -> None:
        """Deprecated back-compat shim. Use `MousePIDController` +
        `set_action` instead. This just points the acceleration toward the
        destination at full magnitude — no PID, no path planning."""
        dx = destination[0] - self.state.location[0]
        dy = destination[1] - self.state.location[1]
        n = (dx * dx + dy * dy) ** 0.5
        if n < 1e-6:
            self.set_action(0.0, 0.0)
        else:
            self.set_action(dx / n, dy / n)

    def stop_navigation(self) -> None:
        self.stop()

    def reset(self):
        Agent.reset(self)
        self.dynamics.ax = 0.0
        self.dynamics.ay = 0.0
        return AgentState(location=self.start_state.location,
                          body_heading=self.start_state.body_heading,
                          velocity=(0.0, 0.0))

    def step(self, delta_t: float) -> None:
        # Agent has no per-step behavior beyond event dispatch; all physics
        # is handled by `Model._point_step`.
        Agent.step(self, delta_t=delta_t)

    def set_view_field(self, view_field: float):
        super().set_view_field(view_field)

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("prey.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 270)
        return rotated_sprite

    @staticmethod
    def create_body_polygon() -> Polygon:
        return Polygon([(.015, 0), (0, 0.005), (-.015, 0), (0, -0.005)])
