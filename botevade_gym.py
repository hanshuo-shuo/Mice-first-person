import sys
import os
import enum
import typing
from collections import deque


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELLWORLD_PATH = os.path.join(BASE_DIR, "cellworld_game-main")
if CELLWORLD_PATH not in sys.path:
    sys.path.insert(0, CELLWORLD_PATH)

# Make absolutely sure we import the local (point-mass) version of
# cellworld_game, not the pip-installed one that may be shadowing it.
for _mod in [m for m in list(sys.modules) if m == "cellworld_game" or m.startswith("cellworld_game.")]:
    del sys.modules[_mod]
import cellworld_game as cwgame
assert os.path.abspath(os.path.dirname(cwgame.__file__)).startswith(CELLWORLD_PATH), \
    f"env3.py imported cellworld_game from {cwgame.__file__} instead of {CELLWORLD_PATH}"
import numpy as np
import math
from gymnasium import Env
from gymnasium import spaces
from enum import Enum
from util import find, normalize_angle, load_cell_ids_near_occlusion
from first_person import FirstPersonVisionWrapper

print("Using env3 (point-mass): (ax, ay) action, velocity in observation")

STACK_FIELDS = [
    "prey_x",
    "prey_y",
    "prey_vx",
    "prey_vy",
    "predator_visible",
    "predator_x",
    "predator_y",
    "predator_direction",
    "near_wall",
    "near_occlusion",
    "time_prey_seen_predator"
]

class Observation(np.ndarray):
    fields = []  # list of field names in the observation

    def __init__(self):
        super().__init__()
        for index, field in enumerate(self.__class__.fields):
            self._create_property(index=index,
                                  field=field)
        self.field_enum = Enum("fields", {field: index for index, field in enumerate(self.__class__.fields)})

    def __new__(cls):
        # Create a new array of zeros with the given shape and dtype
        shape = (len(cls.fields),)
        dtype = np.float32
        buffer = None
        offset = 0
        strides = None
        order = None
        obj = super(Observation, cls).__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.fill(0)
        return obj

    def _create_property(self,
                         index: int,
                         field: str):
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



class Environment(Env):
    def __init__(self):
        self.event_handlers: typing.Dict[str, typing.List[typing.Callable]] = {"reset": [],
                                                                               "step": []}

    def __handle_event__(self, event_name: str, *args):
        for handler in self.event_handlers[event_name]:
            handler(*args)

    def add_event_handler(self, event_name: str, handler: typing.Callable):
        if event_name not in self.event_handlers:
            raise "Event handler not registered"
        self.event_handlers[event_name].append(handler)

    def reset(self,
              options: typing.Optional[dict] = None,
              seed=None):
        self.__handle_event__("reset", options, seed)

    def step(self, action: int):
        self.__handle_event__("step", action)



class BotEvadeObservation(Observation):

    fields = [
        "prey_x",
        "prey_y",
        "prey_vx",
        "prey_vy",
        "predator_visible",
        "predator_x",
        "predator_y",
        "predator_direction",
        "near_wall", #geometric info
        "near_occlusion",
        "time_prey_seen_predator",
        "puffed",
        "puff_cooled_down",
        "finished",
        "prey_goal_distance"
    ]





class BotEvadeEnv(Environment):

    PointOfView = cwgame.BotEvade.PointOfView

    AgentRenderMode = cwgame.Agent.RenderMode

    class ObservationType(enum.Enum):
        DATA = 0
        PIXELS = 1

    class ActionType(enum.Enum):
        DISCRETE = 0
        CONTINUOUS = 1

    def __init__(self,
                 world_name: str,
                 use_lppos: bool,
                 use_predator: bool,
                 max_step: int = 300,
                 reward_function: typing.Callable[[BotEvadeObservation], float] = lambda x: 0,
                 time_step: float = .25,
                 render: bool = False,
                 real_time: bool = False,
                 point_of_view: PointOfView = PointOfView.TOP,
                 agent_render_mode: AgentRenderMode = AgentRenderMode.SPRITE,
                 observation_type: ObservationType = ObservationType.DATA,
                 action_type: ActionType = ActionType.DISCRETE,
                 frame_stack_k: int = 3,
                 prey_max_forward_speed: float = 0.5,
                 prey_max_turning_speed: float = 20.0,
                 predator_prey_forward_speed_ratio: float = 0.15,
                 predator_prey_turning_speed_ratio: float = .175,
                 max_line_of_sight_distance: float = 1.0,
                 predator_prey_line_of_sight_ratio: float = 1.0):

        if observation_type == BotEvadeEnv.ObservationType.PIXELS and not render:
            raise ValueError("Cannot use PIXELS observation type without render")
        self.max_step = max_step
        self.reward_function = reward_function
        self.time_step = time_step
        self.loader = cwgame.CellWorldLoader(world_name=world_name)

        if use_lppos:
            self.action_list = self.loader.tlppo_action_list
        else:
            self.action_list = self.loader.full_action_list

        self.action_type = action_type
        self.frame_stack_k = frame_stack_k
        if self.action_type == BotEvadeEnv.ActionType.DISCRETE:
            # 8 cardinal/diagonal accelerations + stop
            self._discrete_actions = np.array([
                ( 0.0,  0.0),  # stop
                ( 1.0,  0.0), ( 0.707,  0.707),
                ( 0.0,  1.0), (-0.707,  0.707),
                (-1.0,  0.0), (-0.707, -0.707),
                ( 0.0, -1.0), ( 0.707, -0.707),
            ], dtype=np.float32)
            self.action_space = spaces.Discrete(len(self._discrete_actions))
        else:
            # PointMaze-style: (ax, ay) desired acceleration in [-1, 1]
            self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self.model = cwgame.BotEvade(world_name=world_name,
                                     real_time=real_time,
                                     render=render,
                                     use_predator=use_predator,
                                     point_of_view=point_of_view,
                                     agent_render_mode=agent_render_mode,
                                     prey_max_forward_speed=prey_max_forward_speed,
                                     prey_max_turning_speed=prey_max_turning_speed,
                                     predator_prey_forward_speed_ratio=predator_prey_forward_speed_ratio,
                                     predator_prey_turning_speed_ratio=predator_prey_turning_speed_ratio,
                                     max_line_of_sight_distance=max_line_of_sight_distance,
                                     predator_prey_line_of_sight_ratio=predator_prey_line_of_sight_ratio)
        # Save original view field of prey; used for resetting view field of prey
        self.original_prey_view_field = self.model.prey.view_field
        self.observation_type = observation_type
        
        # Load cell ids near occlusion and wall for different worlds
        if world_name == "21_05":
            self.cell_ids_near_occlusion = load_cell_ids_near_occlusion('./data/cell_ids_near_occlusion_21_05.npy')
        elif world_name == "clump01_05":
            self.cell_ids_near_occlusion = load_cell_ids_near_occlusion('./data/cell_ids_near_occlusion.npy')
        else:
            raise ValueError(f"World name {world_name} not supported")
        self.cell_ids_near_wall = load_cell_ids_near_occlusion('./data/cell_ids_near_wall_strict.npy')

        if self.observation_type == BotEvadeEnv.ObservationType.DATA:
            self.observation = BotEvadeObservation()
            self.stack_indices = [
                self.observation.fields.index(field) for field in STACK_FIELDS
            ]
            self.nonstack_indices = [
                idx for idx in range(self.observation.shape[0]) if idx not in self.stack_indices
            ]
            stacked_shape = (len(self.stack_indices) * self.frame_stack_k + len(self.nonstack_indices),)
            self.observation_space = spaces.Box(-np.inf, np.inf, stacked_shape, dtype=np.float32)
            self.frame_stack = deque(maxlen=self.frame_stack_k)
        else:
            self.observation = self.model.view.get_screen(normalized=True)
            self.observation_space = spaces.Box(0.0, 1.0, self.observation.shape, dtype=np.float32)
            self.frame_stack = None
        self.prey_trajectory_length = 0
        self.predator_trajectory_length = 0
        self.episode_reward = 0
        self.step_count = 0
        self.time_prey_seen_predator = -1 # Initialize with -1, meaning never seen
        # info
        self.prey_visible_last_step = 0
        self.predator_visible_last_step = 0

        # Initial values for tracking if prey is near wall and occlusion
        self.near_wall = True #since prey start in the entrance, it is near the wall
        self.near_occlusion = False
        Environment.__init__(self)

    def __update_observation__(self):
        if self.observation_type == BotEvadeEnv.ObservationType.DATA:
            self.observation.prey_x = self.model.prey.state.location[0]
            self.observation.prey_y = self.model.prey.state.location[1]
            self.observation.prey_vx = self.model.prey.state.velocity[0]
            self.observation.prey_vy = self.model.prey.state.velocity[1]
            self.observation.prey_goal_distance = self.model.prey_data.prey_goal_distance


            if self.model.use_predator and self.model.prey_data.predator_visible: 
                self.observation.predator_visible = True
                self.observation.predator_x = self.model.predator.state.location[0]
                self.observation.predator_y = self.model.predator.state.location[1]
                # Normalize direction to the range [0, 2*pi) 
                self.observation.predator_direction = normalize_angle(math.radians(self.model.predator.state.direction)) 
                self.predator_visible_last_step = 1

            else:
                #if predator is not visible, store 0 for predator position and direction
                self.observation.predator_visible = False
                self.observation.predator_x = 0
                self.observation.predator_y = 0
                self.observation.predator_direction = 0
                self.predator_visible_last_step = 0
            # log recent visibility of predator and prey
        
            # Update prey_seen_predator_last_k
            if self.model.use_predator and self.model.prey_data.predator_visible:
                self.time_prey_seen_predator = self.step_count
            self.observation.time_prey_seen_predator = self.time_prey_seen_predator

            # Update predator_seen_prey_last_k
            if self.model.use_predator and self.model.prey_data.prey_visible:
                self.prey_visible_last_step = 1
            else:
                self.prey_visible_last_step = 0

            self.observation.puffed = self.model.prey_data.puffed
            self.observation.puff_cooled_down = self.model.puff_cool_down
            self.observation.finished = not self.model.running
            closest_cell = find(self.loader.locations, self.model.prey.state.location[:2])
            # Check if prey is near wall and occlusion
            self.near_wall = closest_cell in self.cell_ids_near_wall
            self.observation.near_wall = self.near_wall
            self.near_occlusion = closest_cell in self.cell_ids_near_occlusion
            self.observation.near_occlusion = self.near_occlusion
        else:
            self.observation = self.model.view.get_screen()
        return self.__get_stacked_observation__()

    def __get_stacked_observation__(self):
        if self.observation_type != BotEvadeEnv.ObservationType.DATA:
            return self.observation
        current_obs = np.array(self.observation, copy=True)
        current_stack = current_obs[self.stack_indices]
        current_nonstack = current_obs[self.nonstack_indices]
        self.frame_stack.append(current_stack)
        while len(self.frame_stack) < self.frame_stack_k:
            self.frame_stack.appendleft(np.zeros_like(current_stack))
        stacked = np.concatenate(list(self.frame_stack), axis=0)
        return np.concatenate([stacked, current_nonstack], axis=0)


    def set_action(self, action):
        """Action is either a Discrete index (into `_discrete_actions`) or a
        2-vector `(ax, ay) ∈ [-1, 1]²` interpreted as a desired acceleration
        for the prey's point-mass dynamics."""
        if self.action_type == BotEvadeEnv.ActionType.DISCRETE:
            ax, ay = self._discrete_actions[int(action)]
        else:
            ax = float(action[0])
            ay = float(action[1])
        self.model.prey.set_action(ax, ay)

           

    def __step__(self):
        # store previous step visibility before updating observation
        if self.observation_type == BotEvadeEnv.ObservationType.DATA:
            predator_visible_last_step = int(self.predator_visible_last_step)
            prey_visible_last_step = int(self.prey_visible_last_step)
        self.step_count += 1
        truncated = (self.step_count >= self.max_step)
        obs = self.__update_observation__()
        reward = self.reward_function(obs)
        self.episode_reward += reward

        if self.model.prey_data.puffed:
            self.model.prey_data.puffed = False
        if not self.model.running or truncated:
            # add predator position to info for Alex
            survived = 1 if not self.model.running and self.model.prey_data.puff_count == 0 else 0
            info = {"captures": self.model.prey_data.puff_count,
                    "reward": self.episode_reward,
                    "is_success": survived,
                    "survived": survived,
                    "agents": {},
                    "prey_visible_last_step": prey_visible_last_step,
                    "predator_visible_last_step": predator_visible_last_step,
                    "predator_x": self.model.predator.state.location[0],
                    "predator_y": self.model.predator.state.location[1]}

        else:
            info = {
                "prey_visible_last_step": prey_visible_last_step,
                "predator_visible_last_step": predator_visible_last_step,
                "predator_x": self.model.predator.state.location[0],
                "predator_y": self.model.predator.state.location[1]
            }

        return obs, reward, not self.model.running, truncated, info

    def replay_step(self, agents_state: typing.Dict[str, cwgame.AgentState]):
        self.model.set_agents_state(agents_state=agents_state,
                                    delta_t=self.time_step)
        return self.__step__()

    def step(self, action):
        self.set_action(action=action)
        model_t = self.model.time + self.time_step
        while self.model.running and self.model.time < model_t: #while the model is running and the time is less than the model time, step the model
            self.model.step()
        Environment.step(self, action=action)
        return self.__step__()

    def __reset__(self):
        self.near_wall = True
        self.near_occlusion = False
        self.episode_reward = 0
        self.step_count = 0
        self.time_prey_seen_predator = -1
        self.prey_visible_last_step = 0
        self.predator_visible_last_step = 0
        obs = self.__update_observation__()
        if self.observation_type == BotEvadeEnv.ObservationType.DATA:
            self.frame_stack.clear()
            current_obs = np.array(self.observation, copy=True)
            current_stack = current_obs[self.stack_indices]
            for _ in range(self.frame_stack_k):
                self.frame_stack.append(np.array(current_stack, copy=True))
            obs = self.__get_stacked_observation__()
        return obs, {} #initialize the observation in the beginning of the episode


    def reset(self,
              options: typing.Optional[dict] = None,
              seed=None):
        self.model.reset()
        Environment.reset(self, options=options, seed=seed)
        return self.__reset__()

    def replay_reset(self, agents_state: typing.Dict[str, cwgame.AgentState]):
        self.model.reset()
        self.model.set_agents_state(agents_state=agents_state)
        return self.__reset__()

    def close(self):
        self.model.close()
        Env.close(self=self)


class FirstPersonBotEvadeEnv(FirstPersonVisionWrapper):
    """BotEvade with binocular mouse vision and egocentric VLA controls."""

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
        observation_type = kwargs.get("observation_type", BotEvadeEnv.ObservationType.DATA)
        if observation_type != BotEvadeEnv.ObservationType.DATA:
            raise ValueError(
                "FirstPersonBotEvadeEnv owns the pixel observation; "
                "the base observation_type must be DATA",
            )
        kwargs["observation_type"] = BotEvadeEnv.ObservationType.DATA
        kwargs.setdefault("render", False)
        if action_mode != "passthrough":
            kwargs.setdefault("action_type", BotEvadeEnv.ActionType.CONTINUOUS)
        base_env = BotEvadeEnv(*args, **kwargs)
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
