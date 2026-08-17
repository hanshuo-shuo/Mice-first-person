import pygame
import typing
from .resources import Resources
from .util import Point
from .coordinate_converter import CoordinateConverter
from .polygon import Polygon
from .event import EventDispatcher


class AgentState(object):
    def __init__(self,
                 location: Point.type = (0, 0),
                 body_heading: float = 0,
                 velocity: typing.Tuple[float, float] = (0.0, 0.0),
                 *,
                 direction: typing.Optional[float] = None):
        """Physical state for an agent.

        ``body_heading`` is the agent's body orientation in degrees.  It is
        deliberately independent from ``velocity``: a point-mass agent may
        slide sideways without rotating its body.  ``direction`` is accepted
        as a backwards-compatible constructor alias for older callers; the
        canonical stored field is always ``body_heading``.
        """
        if direction is not None:
            body_heading = direction
        self.location = location
        self.body_heading = float(body_heading)
        self.velocity = velocity

    @property
    def direction(self) -> float:
        """Deprecated alias for :attr:`body_heading`.

        Keeping this alias makes old logs and task code readable while
        preventing a second orientation value from being stored.
        """
        return self.body_heading

    @direction.setter
    def direction(self, value: float) -> None:
        self.body_heading = float(value)

    def __iter__(self):
        yield self.location
        yield self.body_heading

    def update(self,
               distance: float,
               rotation: float) -> "AgentState":
        new_heading = self.body_heading + rotation
        return AgentState(location=Point.move(start=self.location,
                                              direction_degrees=new_heading,
                                              distance=distance),
                          body_heading=new_heading,
                          velocity=self.velocity)

    def copy(self) -> "AgentState":
        return AgentState(location=self.location,
                          body_heading=self.body_heading,
                          velocity=self.velocity)


class AgentDynamics(object):
    def __init__(self, forward_speed: float, turn_speed: float):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed

    def __iter__(self):
        yield self.forward_speed
        yield self.turn_speed

    def change(self, delta_t: float) -> tuple:
        return self.forward_speed * delta_t,  self.turn_speed * delta_t


class PointDynamics(object):
    """2D point-mass dynamics: action = (ax, ay) ∈ [-1, 1]²,
    interpreted as desired acceleration scaled by `accel_scale`."""
    def __init__(self,
                 accel_scale: float = 6.0,
                 v_max: float = 0.5,
                 damping: float = 8.0):
        self.ax: float = 0.0
        self.ay: float = 0.0
        self.accel_scale = accel_scale
        self.v_max = v_max
        self.damping = damping


class Agent(EventDispatcher):
    class RenderMode(object):
        SPRITE = 0
        POLYGON = 1

    def __init__(self,
                 view_field: float = 360,
                 collision: bool = True,
                 size: float = .05,
                 sprite_scale: float = 1.0,
                 polygon_color: typing.Tuple[int, int, int] = (0, 80, 120)):
        self.visible = True
        self.render_mode = Agent.RenderMode.SPRITE
        self.view_field = view_field
        self.state: AgentState = AgentState()
        self.dynamics: AgentDynamics = AgentDynamics(forward_speed=0,
                                                     turn_speed=0)
        self._body_polygon = self.create_body_polygon()
        self.body_polygon = None
        self.visibility_polygon: typing.Optional[Polygon] = None
        self.polygon_color = polygon_color
        self.collision = collision

        self.size = size

        self.sprite = None
        self.sprite_scale = sprite_scale
        EventDispatcher.__init__(self, ["reset", "step"])
        self.on_reset = None
        self.on_step = None
        self.on_start = None
        self.name = ""
        self.model = None
        self.running = False
        self.data = None

    def set_view_field(self, view_field: float):
        self.view_field = view_field
        #print(f"Current view field:{self.view_field}")

    def reset(self) -> AgentState:
        self.__dispatch__("reset")
        self.running = True

    def step(self, delta_t: float) -> None:
        self.__dispatch__("step", delta_t)

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("agent.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 90)
        return rotated_sprite

    @staticmethod
    def create_body_polygon() -> Polygon:
        return Polygon.regular((0, 0), .05, 30, sides=6)

    def get_body_polygon(self,
                         state: AgentState = None) -> Polygon:
        # Rotate and then translate the arrow polygon
        if state:
            return self._body_polygon.translate_rotate(
                translation=state.location,
                rotation=state.body_heading,
            )
        else:
            return self._body_polygon.translate_rotate(
                translation=self.state.location,
                rotation=self.state.body_heading,
            )

    def render(self,
               surface: pygame.Surface,
               coordinate_converter: CoordinateConverter):
        if self.visible:
            if self.render_mode == Agent.RenderMode.SPRITE:
                if self.sprite is None:
                    sprite_size = coordinate_converter.scale_from_canonical(self.size) * self.sprite_scale
                    self.sprite = pygame.transform.scale(self.create_sprite(), (sprite_size, sprite_size))
                rotated_sprite = pygame.transform.rotate(self.sprite, self.state.body_heading)
                width, height = rotated_sprite.get_size()
                screen_x, screen_y = coordinate_converter.from_canonical(self.state.location)
                surface.blit(rotated_sprite, (screen_x - width / 2, screen_y - height / 2))
            else:
                self.body_polygon.render(surface=surface,
                                         coordinate_converter=coordinate_converter,
                                         color=self.polygon_color)
