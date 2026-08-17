import copy
import math
import random
import time
import typing
import shapely as sp

from .util import Point
from .agent import Agent, AgentState, PointDynamics
from .visibility import Visibility
from .polygon import Polygon
from .event import EventDispatcher
from .line_of_sight import LineOfSight


class Model(EventDispatcher):

    def __init__(self,
                 world_name: str,
                 arena: Polygon,
                 occlusions: typing.List[Polygon],
                 time_step: float = 0.1,
                 real_time: bool = False,
                 render: bool = False,
                 agent_point_of_view: str = "prey",
                 agent_render_mode: Agent.RenderMode = Agent.RenderMode.SPRITE,
                 max_line_of_sight_distance: float = 1.0,
                 predator_prey_line_of_sight_ratio: float = 1.0):
        self.world_name = world_name
        self.arena = arena
        self.occlusions = occlusions
        self.time_step = time_step
        self.real_time = real_time
        self.render = render
        self.agent_point_of_view = agent_point_of_view
        self.agent_render_mode = agent_render_mode
        self.max_line_of_sight_distance = max_line_of_sight_distance
        # Adjust the line of sight distance for predator and prey based on the ratio
        self.predator_prey_line_of_sight_ratio = predator_prey_line_of_sight_ratio
        self.agents: typing.Dict[str, Agent] = {}
        self.visibility = Visibility(arena=self.arena, occlusions=self.occlusions)
        # Keep simulation randomness on the model instead of relying on the
        # process-global ``random`` module.  Environments can replace/seed
        # this generator at reset time and snapshots can restore it exactly.
        self.rng = random.Random()
        self.last_step = None
        self.time: float = 0
        self.running = False
        self.episode_count = 0
        self.step_count = 0
        self.view: typing.Optional["View"] = None
        self.paused = False
        self.line_of_sight = LineOfSight()
        EventDispatcher.__init__(self, ["before_step",
                                        "after_step",
                                        "before_stop",
                                        "after_stop",
                                        "before_reset",
                                        "after_reset",
                                        "agents_states_update",
                                        "close",
                                        "pause"])
        if self.render:
            self.occlusion_color = (50, 50, 50)
            self.arena_color = (210, 210, 210)
            self.visibility_color = (255, 255, 255)
            from .view import View
            self.view = View(model=self)
            self.view.add_event_handler("quit", self.close)

            def render_occlusions(surface, coordinate_converter):
                for occlusion in self.occlusions:
                    occlusion.render(surface=surface,
                                     coordinate_converter=coordinate_converter,
                                     color=self.occlusion_color)

            def render_arena(surface, coordinate_converter):
                self.arena.render(surface=surface,
                                  coordinate_converter=coordinate_converter,
                                  color=self.arena_color)

            self.view.add_render_step(render_step=render_arena, z_index=0)
            if agent_point_of_view == "":
                self.view.add_render_step(render_step=render_occlusions, z_index=30)
            else:
                self.view.add_render_step(render_step=render_occlusions, z_index=1030)

            self.render_agent_visibility = agent_point_of_view

            def render_visibility(surface, coordinate_converter):
                if self.render_agent_visibility == "":
                    return
                visibility_polygon = self.agents[self.render_agent_visibility].visibility_polygon
                visibility_polygon.render(surface=surface,
                                          coordinate_converter=coordinate_converter,
                                          color=self.visibility_color)

            def render_hidden_area(surface, coordinate_converter):
                import pygame
                if self.render_agent_visibility == "":
                    return
                visibility_polygon = self.agents[self.render_agent_visibility].visibility_polygon
                mask_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                mask_surface.fill((0, 0, 0, 255))
                self.arena.render(surface=mask_surface,
                                  coordinate_converter=coordinate_converter,
                                  color=self.arena_color + (255,))
                visibility_polygon.render(surface=mask_surface,
                                          coordinate_converter=coordinate_converter,
                                          color=(0, 0, 0, 0))
                surface.blit(mask_surface, (0, 0))

            def key_down(key):
                import pygame
                if key == pygame.K_r:
                    if self.agent_render_mode == Agent.RenderMode.SPRITE:
                        self.agent_render_mode = Agent.RenderMode.POLYGON
                    else:
                        self.agent_render_mode = Agent.RenderMode.SPRITE
                    for agent_name, agent in self.agents.items():
                        agent.render_mode = self.agent_render_mode
                elif key == pygame.K_v:
                    agent_names = list(self.agents.keys())
                    if self.render_agent_visibility == "":
                        self.render_agent_visibility = agent_names[0]
                    else:
                        current_agent = agent_names.index(self.render_agent_visibility)
                        if current_agent == len(agent_names) - 1:
                            self.render_agent_visibility = ""
                        else:
                            self.render_agent_visibility = agent_names[current_agent + 1]
                elif key == pygame.K_p:
                    self.pause()
                elif key == pygame.K_n:
                    for agent_name, agent in self.agents.items():
                        if hasattr(agent, "render_path"):
                            agent.render_path = not agent.render_path

            self.view.add_render_step(render_visibility, z_index=20)
            if agent_point_of_view:
                self.view.add_render_step(render_hidden_area, z_index=1000)

            self.view.add_event_handler("key_down", key_down)

    def get_agents_state(self) -> typing.Dict[str, AgentState]:
        agents_state: typing.Dict[str, AgentState] = {}
        for agent_name, agent in self.agents.items():
            agents_state[agent_name] = agent.state.copy()
        return agents_state

    def set_rng(self,
                rng: typing.Optional[random.Random] = None,
                seed: typing.Optional[int] = None) -> None:
        """Attach or seed the model's private Python RNG.

        ``seed`` and ``rng`` are mutually exclusive.  Agents that own
        stochastic behavior (currently ``Robot``) receive the same generator
        object so reset-time placement and navigation choices share one
        reproducible stream.
        """

        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both")
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng.seed(seed)

        for agent in self.agents.values():
            set_agent_rng = getattr(agent, "set_rng", None)
            if set_agent_rng is not None:
                set_agent_rng(self.rng)

    @staticmethod
    def _agent_state_to_dict(state: AgentState) -> dict:
        return {
            "location": tuple(state.location),
            "body_heading": float(state.body_heading),
            "velocity": tuple(state.velocity),
        }

    @staticmethod
    def _agent_state_from_dict(state: dict) -> AgentState:
        # ``direction`` was the old serialized orientation field.  Read it
        # for old snapshots, but never keep it as a second physical truth.
        body_heading = state.get("body_heading", state.get("direction", 0.0))
        return AgentState(
            location=tuple(state["location"]),
            body_heading=float(body_heading),
            velocity=tuple(state.get("velocity", (0.0, 0.0))),
        )

    @staticmethod
    def _point_to_dict(point):
        if point is None:
            return None
        return tuple(point)

    def _get_agent_state_dict(self, agent: Agent) -> dict:
        dynamics_state = {}
        for attribute in (
            "forward_speed",
            "turn_speed",
            "ax",
            "ay",
            "accel_scale",
            "v_max",
            "damping",
        ):
            if hasattr(agent.dynamics, attribute):
                dynamics_state[attribute] = copy.deepcopy(
                    getattr(agent.dynamics, attribute),
                )

        state = {
            "state": self._agent_state_to_dict(agent.state),
            "dynamics": dynamics_state,
            "running": bool(agent.running),
            "visible": bool(agent.visible),
            "view_field": float(agent.view_field),
            "collision": bool(agent.collision),
        }

        # NavigationAgent/Robot state is mutable and affects the next step.
        for attribute in (
            "new_destination",
            "destination",
            "navigation_plan_update_wait",
            "destination_wait",
            "active_navigation",
            "render_path",
            "last_destination_time",
        ):
            if hasattr(agent, attribute):
                value = getattr(agent, attribute)
                if attribute in {"new_destination", "destination"}:
                    value = self._point_to_dict(value)
                state[attribute] = copy.deepcopy(value)
        if hasattr(agent, "path"):
            state["path"] = [self._point_to_dict(point)
                              for point in agent.path]
        if hasattr(agent, "data"):
            state["data"] = copy.deepcopy(agent.data)
        return state

    def _set_agent_state_dict(self, agent: Agent, state: dict) -> None:
        agent.state = self._agent_state_from_dict(state["state"])
        for attribute, value in state.get("dynamics", {}).items():
            if hasattr(agent.dynamics, attribute):
                setattr(agent.dynamics, attribute, copy.deepcopy(value))

        agent.running = bool(state.get("running", agent.running))
        agent.visible = bool(state.get("visible", agent.visible))
        if "view_field" in state:
            agent.view_field = float(state["view_field"])
        if "collision" in state:
            agent.collision = bool(state["collision"])

        for attribute in (
            "navigation_plan_update_wait",
            "destination_wait",
            "active_navigation",
            "render_path",
            "last_destination_time",
        ):
            if attribute in state and hasattr(agent, attribute):
                setattr(agent, attribute, copy.deepcopy(state[attribute]))
        for attribute in ("new_destination", "destination"):
            if attribute in state and hasattr(agent, attribute):
                setattr(agent, attribute, self._point_to_dict(state[attribute]))
        if "path" in state and hasattr(agent, "path"):
            agent.path = [self._point_to_dict(point) for point in state["path"]]
        if "data" in state and hasattr(agent, "data"):
            agent.data = copy.deepcopy(state["data"])

    def get_state_dict(self) -> dict:
        """Return a deep, branch-safe snapshot of the model state.

        The snapshot contains all mutable state needed to continue a model
        from the same point, including agent dynamics, navigation plans and
        the model-local RNG.  Derived geometry is rebuilt on restore.
        """

        return {
            "version": 1,
            "time": float(self.time),
            "step_count": int(self.step_count),
            "episode_count": int(self.episode_count),
            "running": bool(self.running),
            "paused": bool(self.paused),
            "last_step": None if self.last_step is None else float(self.last_step),
            "agents": {
                name: self._get_agent_state_dict(agent)
                for name, agent in self.agents.items()
            },
            "python_rng_state": copy.deepcopy(self.rng.getstate()),
        }

    def set_state_dict(self, state: dict) -> None:
        """Restore a snapshot produced by :meth:`get_state_dict`."""

        if not isinstance(state, dict) or "agents" not in state:
            raise TypeError("state must be a model state dictionary")
        snapshot_agents = set(state["agents"])
        unknown_agents = snapshot_agents - set(self.agents)
        missing_agents = set(self.agents) - snapshot_agents
        if unknown_agents or missing_agents:
            raise ValueError(
                "Snapshot agents do not match model agents: "
                f"unknown={sorted(unknown_agents)}, missing={sorted(missing_agents)}",
            )

        if "python_rng_state" in state:
            self.rng.setstate(copy.deepcopy(state["python_rng_state"]))
        self.set_rng(self.rng)

        for name, agent_state in state["agents"].items():
            self._set_agent_state_dict(self.agents[name], agent_state)

        self.time = float(state.get("time", 0.0))
        self.step_count = int(state.get("step_count", 0))
        self.episode_count = int(state.get("episode_count", self.episode_count))
        self.running = bool(state.get("running", self.running))
        self.paused = bool(state.get("paused", False))
        self.last_step = state.get("last_step")
        if self.last_step is not None:
            self.last_step = float(self.last_step)

        # Rebuild body/visibility geometry and directional line-of-sight from
        # the restored physical states rather than storing derived objects.
        self.set_agents_state(
            agents_state={name: agent.state for name, agent in self.agents.items()},
        )

    def set_agents_state(self,
                         agents_state: typing.Dict[str, AgentState],
                         agents_body_polygons: typing.Dict[str, Polygon] = None,
                         delta_t: float = 0):
        for agent_name, agent_state in agents_state.items():
            agent = self.agents[agent_name]
            agent.state = agent_state
            if agents_body_polygons:
                agent.body_polygon = agents_body_polygons[agent_name]
            else:
                agent.body_polygon = agent.get_body_polygon(state=agent_state)

            agent.visibility_polygon = self.visibility.get_visibility_polygon(src=agent_state.location,
                                                                              direction=agent_state.body_heading,
                                                                              view_field=agent.view_field)

            self.__dispatch__(f"agent_{agent_name}_state_update", agent_state)

        # Check line of sight between each pair of agents (excluding self)
        for agent_name, agent in self.agents.items():
            agent_visibility_polygon = agent.visibility_polygon
            for other_agent_name, other_agent in self.agents.items():
                if other_agent_name == agent_name:
                    continue
                # Determine max line of sight distance
                if agent_name == "predator":
                    max_line_of_sight_distance = self.max_line_of_sight_distance * self.predator_prey_line_of_sight_ratio
                else:
                    max_line_of_sight_distance = self.max_line_of_sight_distance
                distance = Point.distance(agent.state.location, other_agent.state.location)
                if distance <= max_line_of_sight_distance:
                    has_line_of_sight = agent_visibility_polygon.intersects(other_agent.body_polygon)
                else:
                    has_line_of_sight = False
                self.line_of_sight[agent_name, other_agent_name] = has_line_of_sight
        self.__dispatch__("agents_states_update", agents_state, self.line_of_sight)

    def pause(self):
        self.paused = not self.paused
        self.__dispatch__("pause", self)

    def add_agent(self, name: str, agent: Agent):
        agent.name = name
        agent.model = self
        agent.render_mode = self.agent_render_mode
        self.register_event(f"agent_{name}_state_update")
        self.line_of_sight.register_agent(agent=agent)
        self.agents[name] = agent
        if self.render:
            self.view.add_render_step(agent.render, z_index=100)

    def reset(self,
              agents_state: typing.Dict[str, AgentState] = None,
              *,
              seed: typing.Optional[int] = None,
              rng: typing.Optional[random.Random] = None):
        if seed is not None or rng is not None:
            self.set_rng(rng=rng, seed=seed)
        if self.running:
            self.stop()
        self.__dispatch__("before_reset", self)
        self.running = True
        self.episode_count += 1
        self.time = 0.0
        self.step_count = 0
        self.paused = False
        agents_start_state: typing.Dict[str, AgentState] = {}
        agents_body_polygon: typing.Dict[str, Polygon] = {}
        for name, agent in self.agents.items():
            agent_reset_state = agent.reset()
            agent_state = agents_state[name] if agents_state and name in agents_state else agent_reset_state
            agents_start_state[name] = agent_state
            agents_body_polygon[name] = agent.get_body_polygon(state=agent_state)

        self.set_agents_state(agents_state=agents_start_state)
        self.last_step = time.time()
        self.__dispatch__("after_reset", self)

    def stop(self):
        if not self.running:
            return
        self.__dispatch__("before_stop", self)
        self.running = False
        self.__dispatch__("after_stop", self)

    def _point_step(self, agent: Agent, dt: float) -> AgentState:
        """Semi-implicit Euler integration of a 2D point mass with linear
        damping, followed by slide-on-collision. Used for any agent whose
        `dynamics` is a `PointDynamics`."""
        d = agent.dynamics
        vx, vy = agent.state.velocity
        # semi-implicit Euler + linear damping
        vx = vx + (d.accel_scale * d.ax - d.damping * vx) * dt
        vy = vy + (d.accel_scale * d.ay - d.damping * vy) * dt
        # speed cap
        speed = math.hypot(vx, vy)
        if speed > d.v_max:
            s = d.v_max / speed
            vx *= s
            vy *= s

        x0, y0 = agent.state.location
        new_x = x0 + vx * dt
        new_y = y0 + vy * dt

        cand = AgentState(location=(new_x, new_y),
                          body_heading=agent.state.body_heading,
                          velocity=(vx, vy))
        return self._slide_collide(agent, cand, dt)

    def _slide_collide(self,
                       agent: Agent,
                       cand: AgentState,
                       dt: float) -> AgentState:
        """Try full move; if it collides, try sliding along x only, then y
        only; if both fail, stop in place (velocity zeroed). This is the
        point-mass replacement for the unicycle rotate/translate retry."""
        if not agent.collision or self.is_valid_state(
                agent.get_body_polygon(state=cand), agent.collision):
            return cand

        x0, y0 = agent.state.location
        vx, vy = cand.velocity

        # x-only
        x_only = AgentState(location=(x0 + vx * dt, y0),
                            body_heading=agent.state.body_heading,
                            velocity=(vx, 0.0))
        if self.is_valid_state(agent.get_body_polygon(state=x_only),
                               agent.collision):
            return x_only

        # y-only
        y_only = AgentState(location=(x0, y0 + vy * dt),
                            body_heading=agent.state.body_heading,
                            velocity=(0.0, vy))
        if self.is_valid_state(agent.get_body_polygon(state=y_only),
                               agent.collision):
            return y_only

        # fully stuck — stop
        return AgentState(location=(x0, y0),
                          body_heading=agent.state.body_heading,
                          velocity=(0.0, 0.0))

    def is_valid_state(self, agent_polygon: sp.Polygon, collisions: bool) -> bool:
        if not self.arena.contains(agent_polygon):
            return False
        if collisions:
            for occlusion in self.occlusions:
                if agent_polygon.intersects(occlusion):
                    return False
        return True

    def step(self) -> float:
        if not self.running:
            return 0

        if self.paused:
            return 0

        self.__dispatch__("before_step", self)

        if self.real_time:
            while self.last_step + self.time_step > time.time():
                pass

        self.last_step = time.time()
        new_states: typing.Dict[str, AgentState] = {}
        new_body_polygons: typing.Dict[str, Polygon] = {}
        for name, agent in self.agents.items():
            if isinstance(agent.dynamics, PointDynamics):
                new_state = self._point_step(agent, self.time_step)
                agent_polygon = agent.get_body_polygon(state=new_state)
            else:
                dynamics = agent.dynamics
                distance, rotation = dynamics.change(delta_t=self.time_step)
                new_state = agent.state.update(rotation=rotation,
                                               distance=distance)
                agent_polygon = agent.get_body_polygon(state=new_state)
                if not self.is_valid_state(agent_polygon=agent_polygon,
                                           collisions=agent.collision): #try only rotation
                    new_state = agent.state.update(rotation=rotation,
                                                   distance=0)
                    agent_polygon = agent.get_body_polygon(state=new_state)
                    if not self.is_valid_state(agent_polygon=agent_polygon,
                                               collisions=agent.collision): #try only translation
                        new_state = agent.state.update(rotation=0,
                                                       distance=distance)
                        agent_polygon = agent.get_body_polygon(state=new_state)
                        if not self.is_valid_state(agent_polygon=agent_polygon,
                                                   collisions=agent.collision):
                            new_state = agent.state
                            agent_polygon = agent.body_polygon
            new_states[name] = new_state
            new_body_polygons[name] = agent_polygon
        self.set_agents_state(agents_state=new_states, agents_body_polygons=new_body_polygons)
        for name, agent in self.agents.items():
            agent.step(delta_t=self.time_step)
        self.time += self.time_step
        self.step_count += 1
        self.__dispatch__("after_step", self)
        return self.time_step

    def close(self):
        if self.running:
            self.stop()
        self.__dispatch__("close", self)

    def __del__(self):
        self.close()
