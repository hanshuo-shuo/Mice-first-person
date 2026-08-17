import copy
import enum
import os
import random
import sys
import typing
from collections import deque


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELLWORLD_PATH = os.path.join(BASE_DIR, "cellworld_game-main")
# The repository ships the resources required by the default world.  Set the
# cache before importing Cellworld so command-line demos work without network
# access just like the interactive app does.
os.environ.setdefault("CELLWORLD_CACHE", os.path.join(BASE_DIR, "cellworld_cache"))
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

TransitionEvents = typing.Dict[str, typing.Union[bool, float, int]]
RewardTerms = typing.Dict[str, float]

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
              *,
              seed=None,
              options: typing.Optional[dict] = None):
        # Gymnasium owns the standard ``np_random`` initialization.  Keep
        # the event hook after it so handlers observe the seeded environment.
        super().reset(seed=seed)
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
                 reward_function: typing.Callable[[RewardTerms], float] = lambda terms: 0,
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
        self.capture_event_count = 0
        self.goal_event_count = 0
        self.step_count = 0
        self.time_prey_seen_predator = -1 # Initialize with -1, meaning never seen
        self.transition_events: TransitionEvents = self._empty_transition_events()
        self.reward_terms: RewardTerms = self._empty_reward_terms()
        self._step_minimum_distance = 0.0
        self.previous_action = np.zeros((2,), dtype=np.float32)
        # info
        self.prey_visible_last_step = 0
        self.predator_visible_last_step = 0

        # Initial values for tracking if prey is near wall and occlusion
        self.near_wall = True #since prey start in the entrance, it is near the wall
        self.near_occlusion = False
        Environment.__init__(self)

    @staticmethod
    def _empty_transition_events() -> TransitionEvents:
        """Create a fresh, JSON-safe transition-event record."""

        return {
            "puffed": False,
            "capture_event": False,
            "capture_count": 0,
            "predator_sees_prey": False,
            "prey_sees_predator": False,
            "goal_achieved": False,
            "goal_event": False,
            "prey_predator_distance": 0.0,
            "predator_geometric_los": False,
            "predator_in_left_frustum": False,
            "predator_in_right_frustum": False,
            "predator_pixels_visible": False,
            "predator_within_detection_range": False,
            "predator_believed_visible": False,
            "predator_visible_camera": False,
            "predator_visible_geometric": False,
            "minimum_distance": 0.0,
        }

    @staticmethod
    def _empty_reward_terms() -> RewardTerms:
        return {
            "capture": 0.0,
            "goal_achieved": 0.0,
            "goal_distance": 0.0,
            "finished": 0.0,
        }

    def _minimum_predator_distance(self) -> float:
        if not self.model.use_predator:
            return 0.0
        prey_location = self.model.prey.state.location
        predator_location = self.model.predator.state.location
        return float(math.hypot(
            prey_location[0] - predator_location[0],
            prey_location[1] - predator_location[1],
        ))

    def _set_transition_data(
        self,
        *,
        capture_event: typing.Optional[bool] = None,
        goal_event: typing.Optional[bool] = None,
        minimum_distance: typing.Optional[float] = None,
    ) -> None:
        """Snapshot named events and reward terms for the current transition.

        The task has two directional visibility values.  ``predator_sees_prey``
        is the predator-to-prey line of sight, while ``prey_sees_predator`` is
        the reverse direction.  The older ``predator_visible_*`` keys are kept
        as compatibility aliases for consumers that already use them.
        """

        model = self.model
        prey_data = model.prey_data
        if model.use_predator:
            predator_sees_prey = bool(model.line_of_sight["predator", "prey"])
            prey_sees_predator = bool(model.line_of_sight["prey", "predator"])
            prey_sees_predator_geometric = bool(
                model.visibility.line_of_sight(
                    model.prey.state.location,
                    model.predator.state.location,
                )
            )
        else:
            predator_sees_prey = False
            prey_sees_predator = False
            prey_sees_predator_geometric = False

        puffed = bool(prey_data.puffed)
        if capture_event is None:
            capture_event = puffed
        goal_achieved = bool(prey_data.goal_achieved)
        if goal_event is None:
            goal_event = goal_achieved

        goal_distance = float(prey_data.prey_goal_distance)
        prey_predator_distance = self._minimum_predator_distance()
        if minimum_distance is None:
            minimum_distance = prey_predator_distance
        self.transition_events = {
            "puffed": puffed,
            "capture_event": bool(capture_event),
            "capture_count": int(prey_data.puff_count),
            "predator_sees_prey": predator_sees_prey,
            "prey_sees_predator": prey_sees_predator,
            "goal_achieved": goal_achieved,
            "goal_event": bool(goal_event),
            "prey_predator_distance": prey_predator_distance,
            # The base environment has no camera.  The first-person wrapper
            # overwrites these fields after rendering each eye.  Keep the
            # simulator result only as a separate geometric diagnostic.
            "predator_geometric_los": prey_sees_predator_geometric,
            "predator_in_left_frustum": False,
            "predator_in_right_frustum": False,
            "predator_pixels_visible": False,
            "predator_within_detection_range": False,
            "predator_believed_visible": False,
            "predator_visible_camera": False,
            "predator_visible_geometric": prey_sees_predator_geometric,
            "minimum_distance": float(minimum_distance),
        }
        self.reward_terms = {
            "capture": float(self.transition_events["capture_event"]),
            "goal_achieved": float(self.transition_events["goal_event"]),
            "goal_distance": goal_distance,
            "finished": float(not model.running),
        }

    def _base_info(self) -> dict:
        """Return per-transition info with only named event/term values."""

        events = self.transition_events
        info = {
            "transition_events": dict(events),
            "reward_terms": dict(self.reward_terms),
            "puffed": bool(events["puffed"]),
            "capture_event": bool(events["capture_event"]),
            "capture_count": int(events["capture_count"]),
            "cumulative_capture_count": int(events["capture_count"]),
            "capture_event_count": int(self.capture_event_count),
            "predator_sees_prey": bool(events["predator_sees_prey"]),
            "prey_sees_predator": bool(events["prey_sees_predator"]),
            "goal_achieved": bool(events["goal_achieved"]),
            "goal_event": bool(events["goal_event"]),
            "prey_predator_distance": float(events["prey_predator_distance"]),
            "prey_visible_last_step": int(self.prey_visible_last_step),
            "predator_visible_last_step": int(self.predator_visible_last_step),
        }
        for name in (
            "predator_geometric_los",
            "predator_in_left_frustum",
            "predator_in_right_frustum",
            "predator_pixels_visible",
            "predator_within_detection_range",
            "predator_believed_visible",
        ):
            info[name] = bool(events[name])
        info["predator_visible_camera"] = bool(events["predator_visible_camera"])
        info["predator_visible_geometric"] = bool(events["predator_visible_geometric"])
        if self.model.use_predator:
            info["predator_x"] = self.model.predator.state.location[0]
            info["predator_y"] = self.model.predator.state.location[1]
        return info

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
                self.observation.predator_direction = normalize_angle(
                    math.radians(self.model.predator.state.body_heading),
                )
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
            self.previous_action = np.asarray(int(action), dtype=np.int64)
        else:
            ax = float(action[0])
            ay = float(action[1])
            self.previous_action = np.asarray(action, dtype=np.float32).copy()
        self.model.prey.set_action(ax, ay)

           

    def __step__(self):
        # store previous step visibility before updating observation
        predator_visible_last_step = int(self.predator_visible_last_step)
        prey_visible_last_step = int(self.prey_visible_last_step)
        self.step_count += 1
        truncated = (self.step_count >= self.max_step)
        obs = self.__update_observation__()
        self._set_transition_data(minimum_distance=self._step_minimum_distance)
        self.capture_event_count += int(self.transition_events["capture_event"])
        self.goal_event_count += int(self.transition_events["goal_event"])
        reward = float(self.reward_function(self.reward_terms))
        self.episode_reward += reward

        if self.model.prey_data.puffed:
            self.model.prey_data.puffed = False
        if not self.model.running or truncated:
            # add predator position to info for Alex
            survived = 1 if not self.model.running and self.model.prey_data.puff_count == 0 else 0
            info = self._base_info()
            info.update({
                "captures": self.model.prey_data.puff_count,
                "reward": self.episode_reward,
                "is_success": survived,
                "survived": survived,
                "agents": {},
                "capture_event_count": self.capture_event_count,
                "goal_event_count": self.goal_event_count,
                "episode_metrics": {
                    "capture_count": int(self.model.prey_data.puff_count),
                    "goal_count": int(self.goal_event_count),
                    "survived": bool(survived),
                },
            })
            info["prey_visible_last_step"] = prey_visible_last_step
            info["predator_visible_last_step"] = predator_visible_last_step

        else:
            info = self._base_info()
            # These fields describe visibility at the beginning of this
            # transition and are retained for existing downstream consumers.
            info["prey_visible_last_step"] = prey_visible_last_step
            info["predator_visible_last_step"] = predator_visible_last_step

        return obs, reward, not self.model.running, truncated, info

    def replay_step(self, agents_state: typing.Dict[str, cwgame.AgentState]):
        self._step_minimum_distance = self._minimum_predator_distance()
        self.model.set_agents_state(agents_state=agents_state,
                                    delta_t=self.time_step)
        self._step_minimum_distance = min(
            self._step_minimum_distance,
            self._minimum_predator_distance(),
        )
        return self.__step__()

    def step(self, action):
        self.set_action(action=action)
        self._step_minimum_distance = self._minimum_predator_distance()
        model_t = self.model.time + self.time_step
        while self.model.running and self.model.time < model_t: #while the model is running and the time is less than the model time, step the model
            self.model.step()
            self._step_minimum_distance = min(
                self._step_minimum_distance,
                self._minimum_predator_distance(),
            )
        Environment.step(self, action=action)
        return self.__step__()

    def __reset__(self):
        self.near_wall = True
        self.near_occlusion = False
        self.episode_reward = 0
        self.capture_event_count = 0
        self.goal_event_count = 0
        self.step_count = 0
        self.time_prey_seen_predator = -1
        self.prey_visible_last_step = 0
        self.predator_visible_last_step = 0
        self._step_minimum_distance = self._minimum_predator_distance()
        if self.action_type == BotEvadeEnv.ActionType.DISCRETE:
            self.previous_action = np.asarray(0, dtype=np.int64)
        else:
            self.previous_action = np.zeros((2,), dtype=np.float32)
        obs = self.__update_observation__()
        if self.observation_type == BotEvadeEnv.ObservationType.DATA:
            self.frame_stack.clear()
            current_obs = np.array(self.observation, copy=True)
            current_stack = current_obs[self.stack_indices]
            for _ in range(self.frame_stack_k):
                self.frame_stack.append(np.array(current_stack, copy=True))
            obs = self.__get_stacked_observation__()
        self._set_transition_data(capture_event=False, goal_event=False)
        return obs, {
            "transition_events": dict(self.transition_events),
            "reward_terms": dict(self.reward_terms),
        } #initialize the observation in the beginning of the episode

    def _gym_rng_state(self) -> dict:
        np_random = getattr(self, "_np_random", None)
        if np_random is None:
            np_random = self.np_random
        return {
            "bit_generator": np_random.bit_generator.__class__.__name__,
            "state": copy.deepcopy(np_random.bit_generator.state),
        }

    def _restore_gym_rng_state(self, state: dict) -> None:
        if not state:
            return
        bit_generator_name = state.get("bit_generator", "PCG64")
        bit_generator_type = getattr(np.random, bit_generator_name, None)
        if bit_generator_type is None:
            raise ValueError(f"Unsupported NumPy bit generator: {bit_generator_name}")
        current = getattr(self, "_np_random", None)
        if current is None or current.bit_generator.__class__.__name__ != bit_generator_name:
            self._np_random = np.random.Generator(bit_generator_type())
        self._np_random.bit_generator.state = copy.deepcopy(state["state"])

    def get_state_dict(self) -> dict:
        """Return a complete, branch-safe environment snapshot.

        Besides the physical model, this stores task counters, frame-stack
        history, event/reward bookkeeping and all RNG streams used by the
        environment.  A snapshot can therefore be restored before running a
        different policy and then restored again for a paired counterfactual.
        """

        frame_stack = None
        if self.frame_stack is not None:
            frame_stack = [np.array(frame, copy=True) for frame in self.frame_stack]

        environment_state = {
            "prey_trajectory_length": copy.deepcopy(self.prey_trajectory_length),
            "predator_trajectory_length": copy.deepcopy(self.predator_trajectory_length),
            "episode_reward": float(self.episode_reward),
            "capture_event_count": int(self.capture_event_count),
            "goal_event_count": int(self.goal_event_count),
            "step_count": int(self.step_count),
            "time_prey_seen_predator": int(self.time_prey_seen_predator),
            "transition_events": copy.deepcopy(self.transition_events),
            "reward_terms": copy.deepcopy(self.reward_terms),
            "step_minimum_distance": float(self._step_minimum_distance),
            "previous_action": copy.deepcopy(self.previous_action),
            "prey_visible_last_step": int(self.prey_visible_last_step),
            "predator_visible_last_step": int(self.predator_visible_last_step),
            "near_wall": bool(self.near_wall),
            "near_occlusion": bool(self.near_occlusion),
            "observation": np.array(self.observation, copy=True),
            "frame_stack": frame_stack,
        }

        return {
            "version": 1,
            "model": self.model.get_state_dict(),
            "environment": environment_state,
            "rng": {
                # These global streams are not used for simulation choices,
                # but preserving them makes policy-side random behavior
                # branch-safe as well.
                "python": copy.deepcopy(random.getstate()),
                "numpy": copy.deepcopy(np.random.get_state()),
                "gymnasium": self._gym_rng_state(),
            },
        }

    def set_state_dict(self, state: dict) -> None:
        """Restore a snapshot produced by :meth:`get_state_dict`."""

        if not isinstance(state, dict) or "model" not in state:
            raise TypeError("state must be an environment state dictionary")
        self.model.set_state_dict(state["model"])

        environment_state = state.get("environment", {})
        for attribute in (
            "prey_trajectory_length",
            "predator_trajectory_length",
            "episode_reward",
            "capture_event_count",
            "goal_event_count",
            "step_count",
            "time_prey_seen_predator",
            "prey_visible_last_step",
            "predator_visible_last_step",
            "near_wall",
            "near_occlusion",
        ):
            if attribute not in environment_state:
                continue
            setattr(self, attribute, copy.deepcopy(environment_state[attribute]))
        if "step_minimum_distance" in environment_state:
            self._step_minimum_distance = float(environment_state["step_minimum_distance"])
        if "transition_events" in environment_state:
            self.transition_events = copy.deepcopy(environment_state["transition_events"])
        if "reward_terms" in environment_state:
            self.reward_terms = copy.deepcopy(environment_state["reward_terms"])
        if "previous_action" in environment_state:
            self.previous_action = copy.deepcopy(environment_state["previous_action"])

        if "observation" in environment_state:
            observation = np.asarray(environment_state["observation"])
            if self.observation_type == BotEvadeEnv.ObservationType.DATA:
                if self.observation.shape != observation.shape:
                    raise ValueError("Snapshot observation shape does not match this environment")
                self.observation[...] = observation
            else:
                self.observation = np.array(observation, copy=True)

        frame_stack = environment_state.get("frame_stack")
        if self.frame_stack is not None:
            self.frame_stack.clear()
            if frame_stack is not None:
                for frame in frame_stack:
                    self.frame_stack.append(np.array(frame, copy=True))

        rng_state = state.get("rng", {})
        if "gymnasium" in rng_state:
            self._restore_gym_rng_state(rng_state["gymnasium"])
        if "python" in rng_state:
            random.setstate(copy.deepcopy(rng_state["python"]))
        if "numpy" in rng_state:
            np.random.set_state(copy.deepcopy(rng_state["numpy"]))


    def reset(self,
              *,
              seed=None,
              options: typing.Optional[dict] = None):
        # This must happen before model.reset(): the model's private RNG is
        # derived from Gymnasium's generator and is then consumed by predator
        # placement and the first roaming destination.
        Environment.reset(self, options=options, seed=seed)
        if seed is not None:
            model_seed = int(self.np_random.integers(0, 2**63 - 1))
            self.model.set_rng(seed=model_seed)
        self.model.reset()
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
        vision_far_clip: float = 2.0,
        vision_detection_range: typing.Optional[float] = None,
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
            far_clip=vision_far_clip,
            detection_range=vision_detection_range,
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
