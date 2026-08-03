import sys
import os
import enum
import typing
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELLWORLD_PATH = os.path.join(BASE_DIR, "cellworld_game-main")
if CELLWORLD_PATH not in sys.path:
    sys.path.insert(0, CELLWORLD_PATH)

for _mod in [m for m in list(sys.modules) if m == "cellworld_game" or m.startswith("cellworld_game.")]:
    del sys.modules[_mod]
import cellworld_game as cwgame
assert os.path.abspath(os.path.dirname(cwgame.__file__)).startswith(CELLWORLD_PATH), \
    f"oasis_env.py imported cellworld_game from {cwgame.__file__} instead of {CELLWORLD_PATH}"

import numpy as np
import math
from gymnasium import Env, spaces
from enum import Enum
from util import normalize_angle
from first_person import FirstPersonVisionWrapper

print("Using oasis_gym (point-mass): (ax, ay) action, velocity in observation")

# Fields that get stacked across frames
STACK_FIELDS = [
    "prey_x",
    "prey_y",
    "prey_vx",
    "prey_vy",
    "predator_visible",
    "predator_x",
    "predator_y",
    "predator_direction",
    "time_prey_seen_predator",
    "goal_x",
    "goal_y",
]


class Observation(np.ndarray):
    fields = []

    def __init__(self):
        super().__init__()
        for index, field in enumerate(self.__class__.fields):
            self._create_property(index=index, field=field)
        self.field_enum = Enum("fields", {field: index for index, field in enumerate(self.__class__.fields)})

    def __new__(cls):
        shape = (len(cls.fields),)
        obj = super(Observation, cls).__new__(cls, shape, np.float32, None, 0, None, None)
        obj.fill(0)
        return obj

    def _create_property(self, index: int, field: str):
        def getter(self):
            return self[index]
        def setter(self, value):
            self[index] = value
        setattr(self.__class__, field, property(getter, setter))

    def __setitem__(self, field: typing.Union[Enum, int], value):
        if isinstance(field, Enum):
            np.ndarray.__setitem__(self, field.value, value)
        else:
            np.ndarray.__setitem__(self, field, value)

    def __getitem__(self, field: typing.Union[Enum, int]) -> np.ndarray:
        if isinstance(field, Enum):
            return np.ndarray.__getitem__(self, field.value)
        else:
            return np.ndarray.__getitem__(self, field)


class OasisObservation(Observation):
    fields = [
        "prey_x",
        "prey_y",
        "prey_vx",
        "prey_vy",
        "predator_visible",
        "predator_x",
        "predator_y",
        "predator_direction",
        "time_prey_seen_predator",
        "goal_x",           # current goal x
        "goal_y",           # current goal y
        "goals_remaining",  # how many goals are left in the sequence
        "puffed",
        "puff_cooled_down",
        "finished",
        "prey_goal_distance",
    ]


class Environment(Env):
    def __init__(self):
        self.event_handlers: typing.Dict[str, typing.List[typing.Callable]] = {"reset": [], "step": []}

    def __handle_event__(self, event_name: str, *args):
        for handler in self.event_handlers[event_name]:
            handler(*args)

    def add_event_handler(self, event_name: str, handler: typing.Callable):
        if event_name not in self.event_handlers:
            raise ValueError("Event handler not registered")
        self.event_handlers[event_name].append(handler)

    def reset(self, options: typing.Optional[dict] = None, seed=None):
        self.__handle_event__("reset", options, seed)

    def step(self, action):
        self.__handle_event__("step", action)


class OasisEnv(Environment):

    PointOfView = cwgame.Oasis.PointOfView

    AgentRenderMode = cwgame.Agent.RenderMode

    class ActionType(enum.Enum):
        DISCRETE = 0
        CONTINUOUS = 1

    DEFAULT_GOAL_LOCATIONS = [
        (0.265625, 0.5),
        (0.3125, 0.7435696448143734),
        (0.3125, 0.1752404735808355),
        (0.4765625, 0.45940505919760444),
        (0.640625, 0.7435696448143734),
        (0.6875, 0.1752404735808355),
        (0.78125, 0.5),
    ]

    def __init__(
        self,
        world_name: str = "21_05",
        goal_locations: typing.Optional[typing.List[typing.Tuple[float, float]]] = None,
        goal_sequence_generator: typing.Optional[typing.Callable] = None,
        use_predator: bool = True,
        max_step: int = 600,
        reward_function: typing.Callable[[OasisObservation], float] = lambda x: 0,
        time_step: float = .25,
        render: bool = False,
        real_time: bool = False,
        point_of_view: PointOfView = PointOfView.TOP,
        agent_render_mode: AgentRenderMode = AgentRenderMode.SPRITE,
        action_type: ActionType = ActionType.CONTINUOUS,
        frame_stack_k: int = 3,
        puff_cool_down_time: float = .5,
        puff_threshold: float = .1,
        goal_threshold: float = .025,
        goal_time: float = 1.0,
        prey_max_forward_speed: float = 0.5,
        prey_max_turning_speed: float = 20.0,
        predator_prey_forward_speed_ratio: float = 0.15,
        predator_prey_turning_speed_ratio: float = 0.175,
        max_line_of_sight_distance: float = 1.0,
    ):
        if goal_locations is None:
            goal_locations = self.DEFAULT_GOAL_LOCATIONS

        self.max_step = max_step
        self.reward_function = reward_function
        self.time_step = time_step
        self.loader = cwgame.CellWorldLoader(world_name=world_name)
        self.action_type = action_type
        self.frame_stack_k = frame_stack_k

        if self.action_type == OasisEnv.ActionType.DISCRETE:
            self._discrete_actions = np.array([
                ( 0.0,  0.0),
                ( 1.0,  0.0), ( 0.707,  0.707),
                ( 0.0,  1.0), (-0.707,  0.707),
                (-1.0,  0.0), (-0.707, -0.707),
                ( 0.0, -1.0), ( 0.707, -0.707),
            ], dtype=np.float32)
            self.action_space = spaces.Discrete(len(self._discrete_actions))
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self.model = cwgame.Oasis(
            world_name=world_name,
            goal_locations=goal_locations,
            goal_sequence_generator=goal_sequence_generator,
            use_predator=use_predator,
            puff_cool_down_time=puff_cool_down_time,
            puff_threshold=puff_threshold,
            goal_threshold=goal_threshold,
            goal_time=goal_time,
            time_step=0.025,
            real_time=real_time,
            render=render,
            point_of_view=point_of_view,
            agent_render_mode=agent_render_mode,
            prey_max_forward_speed=prey_max_forward_speed,
            prey_max_turning_speed=prey_max_turning_speed,
            predator_prey_forward_speed_ratio=predator_prey_forward_speed_ratio,
            predator_prey_turning_speed_ratio=predator_prey_turning_speed_ratio,
            max_line_of_sight_distance=max_line_of_sight_distance,
        )

        self.original_prey_view_field = self.model.prey.view_field

        self.observation = OasisObservation()
        self.stack_indices = [
            self.observation.fields.index(field) for field in STACK_FIELDS
        ]
        self.nonstack_indices = [
            idx for idx in range(self.observation.shape[0]) if idx not in self.stack_indices
        ]
        stacked_shape = (len(self.stack_indices) * self.frame_stack_k + len(self.nonstack_indices),)
        self.observation_space = spaces.Box(-np.inf, np.inf, stacked_shape, dtype=np.float32)
        self.frame_stack = deque(maxlen=self.frame_stack_k)

        self.episode_reward = 0
        self.step_count = 0
        self.time_prey_seen_predator = -1
        self.predator_visible_last_step = 0
        self.prey_visible_last_step = 0

        Environment.__init__(self)

    def _compute_predator_visible(self) -> bool:
        """Check if prey can see predator via line of sight."""
        if not self.model.use_predator:
            return False
        return self.model.visibility.line_of_sight(
            self.model.prey.state.location,
            self.model.predator.state.location,
        )

    def _compute_prey_visible(self) -> bool:
        """Check if predator can see prey via line of sight."""
        if not self.model.use_predator:
            return False
        return self.model.visibility.line_of_sight(
            self.model.predator.state.location,
            self.model.prey.state.location,
        )

    def __update_observation__(self):
        obs = self.observation
        obs.prey_x = self.model.prey.state.location[0]
        obs.prey_y = self.model.prey.state.location[1]
        obs.prey_vx = self.model.prey.state.velocity[0]
        obs.prey_vy = self.model.prey.state.velocity[1]
        obs.prey_goal_distance = self.model.prey_goal_distance

        # Current goal
        if self.model.goal_location is not None:
            obs.goal_x = self.model.goal_location[0]
            obs.goal_y = self.model.goal_location[1]
        else:
            obs.goal_x = 0.0
            obs.goal_y = 0.0
        obs.goals_remaining = float(len(self.model.goal_sequence))

        # Predator visibility
        predator_visible = self._compute_predator_visible()
        if predator_visible:
            obs.predator_visible = True
            obs.predator_x = self.model.predator.state.location[0]
            obs.predator_y = self.model.predator.state.location[1]
            obs.predator_direction = normalize_angle(math.radians(self.model.predator.state.direction))
            self.predator_visible_last_step = 1
        else:
            obs.predator_visible = False
            obs.predator_x = 0.0
            obs.predator_y = 0.0
            obs.predator_direction = 0.0
            self.predator_visible_last_step = 0

        if predator_visible:
            self.time_prey_seen_predator = self.step_count
        obs.time_prey_seen_predator = self.time_prey_seen_predator

        prey_visible = self._compute_prey_visible()
        self.prey_visible_last_step = 1 if prey_visible else 0

        obs.puffed = self.model.puffed
        obs.puff_cooled_down = self.model.puff_cool_down
        obs.finished = not self.model.running

        return self.__get_stacked_observation__()

    def __get_stacked_observation__(self):
        current_obs = np.array(self.observation, copy=True)
        current_stack = current_obs[self.stack_indices]
        current_nonstack = current_obs[self.nonstack_indices]
        self.frame_stack.append(current_stack)
        while len(self.frame_stack) < self.frame_stack_k:
            self.frame_stack.appendleft(np.zeros_like(current_stack))
        stacked = np.concatenate(list(self.frame_stack), axis=0)
        return np.concatenate([stacked, current_nonstack], axis=0)

    def set_action(self, action):
        """Map gym action to prey (ax, ay) acceleration."""
        if self.action_type == OasisEnv.ActionType.DISCRETE:
            ax, ay = self._discrete_actions[int(action)]
        else:
            ax = float(action[0])
            ay = float(action[1])
        self.model.prey.set_action(ax, ay)

    def __step__(self):
        predator_visible_last_step = int(self.predator_visible_last_step)
        prey_visible_last_step = int(self.prey_visible_last_step)

        self.step_count += 1
        truncated = (self.step_count >= self.max_step)
        obs = self.__update_observation__()
        reward = self.reward_function(obs)
        self.episode_reward += reward

        if self.model.puffed:
            self.model.puffed = False

        done = not self.model.running
        if done or truncated:
            survived = 1 if done and self.model.puff_count == 0 else 0
            info = {
                "captures": self.model.puff_count,
                "reward": self.episode_reward,
                "is_success": survived,
                "survived": survived,
                "agents": {},
                "prey_visible_last_step": prey_visible_last_step,
                "predator_visible_last_step": predator_visible_last_step,
            }
            if self.model.use_predator:
                info["predator_x"] = self.model.predator.state.location[0]
                info["predator_y"] = self.model.predator.state.location[1]
        else:
            info = {
                "prey_visible_last_step": prey_visible_last_step,
                "predator_visible_last_step": predator_visible_last_step,
            }
            if self.model.use_predator:
                info["predator_x"] = self.model.predator.state.location[0]
                info["predator_y"] = self.model.predator.state.location[1]

        return obs, reward, done, truncated, info

    def step(self, action):
        self.set_action(action=action)
        model_t = self.model.time + self.time_step
        while self.model.running and self.model.time < model_t:
            self.model.step()
        Environment.step(self, action=action)
        return self.__step__()

    def __reset__(self):
        self.episode_reward = 0
        self.step_count = 0
        self.time_prey_seen_predator = -1
        self.predator_visible_last_step = 0
        self.prey_visible_last_step = 0

        self.frame_stack.clear()
        obs = self.__update_observation__()
        current_obs = np.array(self.observation, copy=True)
        current_stack = current_obs[self.stack_indices]
        for _ in range(self.frame_stack_k):
            self.frame_stack.append(np.array(current_stack, copy=True))
        obs = self.__get_stacked_observation__()
        return obs, {}

    def reset(self, options: typing.Optional[dict] = None, seed=None):
        self.model.reset()
        Environment.reset(self, options=options, seed=seed)
        return self.__reset__()

    def close(self):
        self.model.close()
        Env.close(self=self)


class FirstPersonOasisEnv(FirstPersonVisionWrapper):
    """Oasis with binocular mouse vision and egocentric VLA controls."""

    def __init__(
        self,
        *args,
        vision_width: int = 192,
        vision_height: int = 128,
        vision_fov: float = 120.0,
        vision_camera_height: float = 0.025,
        vision_eye_yaw: float = 40.0,
        vision_eye_separation: float = 0.016,
        vision_eye_forward_offset: float = 0.012,
        observation_mode: str = "mouse",
        action_mode: str = "egocentric_velocity",
        max_body_turn_rate: float = 180.0,
        max_head_turn_rate: float = 240.0,
        head_yaw_limit: float = 60.0,
        head_recenter_rate: float = 90.0,
        velocity_gain: float = 5.0,
        render_mode: str = "rgb_array",
        **kwargs,
    ):
        kwargs.setdefault("render", False)
        if action_mode != "passthrough":
            kwargs.setdefault("action_type", OasisEnv.ActionType.CONTINUOUS)
        base_env = OasisEnv(*args, **kwargs)
        super().__init__(
            base_env,
            width=vision_width,
            height=vision_height,
            horizontal_fov=vision_fov,
            camera_height=vision_camera_height,
            eye_yaw_degrees=vision_eye_yaw,
            eye_separation=vision_eye_separation,
            eye_forward_offset=vision_eye_forward_offset,
            observation_mode=observation_mode,
            action_mode=action_mode,
            max_body_turn_rate=max_body_turn_rate,
            max_head_turn_rate=max_head_turn_rate,
            head_yaw_limit=head_yaw_limit,
            head_recenter_rate=head_recenter_rate,
            velocity_gain=velocity_gain,
            render_mode=render_mode,
        )
