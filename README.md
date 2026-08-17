## Cellworld Game — Single-Prey RL (BotEvade / Oasis)

Base repo: [germanespinosa/cellworld_game](https://github.com/germanespinosa/cellworld_game)

---

# Cellworld Gymnasium Environments

Base repo: [https://github.com/germanespinosa/cellworld_game](https://github.com/germanespinosa/cellworld_game)

Two Gymnasium-compatible environments wrapping the Cellworld simulation:


| Environment   | File              | Task                                                               |
| ------------- | ----------------- | ------------------------------------------------------------------ |
| `BotEvadeEnv` | `botevade_gym.py` | Prey evades a predator robot to reach a single goal                |
| `OasisEnv`    | `oasis_gym.py`    | Prey visits a sequence of goal locations while avoiding a predator |


---

## Environment Description

Both environments are **POMDPs**: the prey agent does not always have line-of-sight to the predator, so the observation is only a partial view of the true state. This has practical implications for algorithm choice:

- **Model-free methods** (e.g., SAC, PPO) work fine out of the box; frame stacking (`frame_stack_k`) provides a basic temporal context.
- **Model-based methods** should account for partial observability — a recurrent world model (e.g., LSTM-based) is recommended to maintain a belief state over the hidden predator position.

## RL Resources


| Library           | Link                                                                                                                             | Notes                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Spinning Up       | [https://spinningup.openai.com/en/latest/user/introduction.html](https://spinningup.openai.com/en/latest/user/introduction.html) | Good conceptual intro to RL algorithms                               |
| Stable-Baselines3 | [https://stable-baselines3.readthedocs.io/en/master/index.html](https://stable-baselines3.readthedocs.io/en/master/index.html)   | Easy to use; covers standard model-free algorithms                   |
| Tianshou          | [https://tianshou.org/en/stable/](https://tianshou.org/en/stable/)                                                               | More flexible; supports n-step returns and custom collectors         |
| SheepRL           | [https://github.com/Eclectic-Sheep/sheeprl](https://github.com/Eclectic-Sheep/sheeprl)                                           | Model-based RL; includes Dreamer-V3 with a built-in LSTM world model |


## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate Mice-BotEvade
```

### 2. Install the cellworld_game package

The simulation lives in the bundled `cellworld_game-main/` folder.
No extra install step is needed — both gym files add it to `sys.path` automatically.

### 3. Verify the install

```bash
python -c "import cellworld_game; print('OK')"
```

---

### Design notes

#### Prey dynamics: PointMaze-style (ax, ay)

Prey use 2D point-mass dynamics (`(ax, ay)` action, `(vx, vy)` state) matching `gymnasium-robotics/PointMaze` — semi-implicit Euler + linear damping. The old unicycle + A\* + PID (`set_destination`) is replaced; predator (`Robot`) keeps unicycle navigation.

`AgentState.body_heading` is the single physical body orientation, stored independently from velocity so sideways sliding does not rotate the agent. First-person yaw actions update this field; collision geometry, simulator visibility, top-down rendering, camera rays, and proprioception all read it. Head yaw remains a separate relative gaze state in the first-person wrapper.

#### Mouse first-person observations for VLM/VLA policies

`FirstPersonBotEvadeEnv` and `FirstPersonOasisEnv` expose low, binocular, mouse-centred vision. Each eye has a 120° horizontal field of view and is turned 40° outward, giving a wide combined visual field with a frontal overlap. Both eyes use the same arena and occlusion polygons as physics, so visual and physical occlusion stay aligned.

The recommended VLA observation is a Gymnasium `Dict`:

```python
{
    "image_left":     uint8[128, 192, 3],
    "image_right":    uint8[128, 192, 3],
    "proprio":        float32[3],  # forward speed, body yaw rate, head yaw
    "previous_action": float32[2],
}
```

The default action is normalized body-frame control `[forward_velocity, body_yaw_rate]` in `[-1, 1]²`. A low-level adapter maps the desired forward velocity to the existing world-frame point-mass acceleration; the base environment still owns integration, damping, collision, reward, and termination. Consequently, the first action dimension always means “forward in the current image/body frame,” rather than world `x`.

```python
import numpy as np
from botevade_gym import BotEvadeEnv, FirstPersonBotEvadeEnv

env = FirstPersonBotEvadeEnv(
    world_name="21_05",
    use_lppos=False,
    use_predator=True,
    action_type=BotEvadeEnv.ActionType.CONTINUOUS,
    vision_width=192,       # per eye
    vision_height=128,
    vision_fov=120.0,       # per eye
    vision_far_clip=2.0,
    vision_detection_range=2.0,
    vision_eye_yaw=40.0,    # outward yaw per eye
    observation_mode="mouse",
    action_mode="egocentric_velocity",
    render_mode="rgb_array",
)

obs, info = env.reset()
assert env.observation_space.contains(obs)
assert obs["image_left"].shape == (128, 192, 3)

# Move forward at 50% speed while turning right at 20% yaw rate.
action = np.array([0.5, -0.2], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)

frame = env.render()  # side-by-side binocular preview
env.close()
```

For active looking/peeking, select `action_mode="egocentric_velocity_head"`. The action becomes `[forward_velocity, body_yaw_rate, head_yaw_rate]`; head yaw is limited to ±60° and recentres when the third command returns to zero.

For an existing `BotEvadeEnv` or `OasisEnv`, the equivalent generic wrapper is:

```python
from first_person import FirstPersonVisionWrapper

env = FirstPersonVisionWrapper(base_env, width=192, height=128)
```

For a strict legacy ablation, use `observation_mode="single_rgb", action_mode="passthrough"`; this keeps one centred RGB image and the original world-frame action. The legacy top-down renderer is also preserved: construct the base environment with `render=True`, then call `env.render_top_down()`.

Runnable VLA contracts:

```bash
# Two-eye observation, [forward velocity, body yaw rate]
conda run -n Mice-BotEvade python -B vlm_first_person_demo.py --human

# Add independent head/gaze control
conda run -n Mice-BotEvade python -B vlm_first_person_demo.py --active-gaze --human
```

To run the dynamics-respecting cell-path controller and record a split-screen solution GIF (first-person view plus a progressively drawn top-down trajectory):

```bash
conda run -n Mice-BotEvade python -B solve_first_person.py \
  --output results/botevade_first_person_solution.gif
```

Add `--human` to display the same frames in a live window while recording. The transparent path solver deliberately uses `action_mode="passthrough"` because it is an old world-frame baseline; learned VLA policies use the egocentric default. The solver never writes location/velocity or bypasses collision physics.

The same run also writes `results/botevade_first_person_solution_trajectory.png`, containing the final top-down prey and predator trajectories.

#### Playable macOS app and human-demonstration collection

Open `Mouse First-Person Lab.app` in Finder to launch the local game. The app has two start buttons:

- **开始试玩** — play without saving data. Press `R` at any time to begin recording.
- **开始并采集数据** — enter the same game with recording enabled immediately.

Keyboard controls:

| Key | Control |
| --- | --- |
| `↑` / `↓` | Forward / backward velocity |
| `←` / `→` | Turn the mouse body left / right |
| `A` / `D` | Look left / right without translating |
| `Space` | Stop translation |
| `R` | Start or stop data recording |
| `N` | Save the current segment and start a new episode |
| `P` | Pause / resume |
| `M` or `Esc` | Return to the main menu |
| `Q` | Quit |

Recorded sessions are written to `datasets/human_demos/session_<timestamp>/`. Every file contains at most 60 seconds at 10 Hz, so long sessions are saved incrementally:

```text
session_<timestamp>/
├── session.json          # environment and field schemas
├── episode_00000.npz     # compressed transitions, no pickle objects
├── episode_00000.json    # length, return, success and end reason
└── ...
```

Each NPZ stores aligned `observation_t → action_t → reward_t/done_t` arrays:

```python
import numpy as np

episode = np.load("episode_00000.npz", allow_pickle=False)
left_images = episode["image_left"]       # (T, 128, 192, 3), uint8
right_images = episode["image_right"]     # (T, 128, 192, 3), uint8
proprio = episode["proprio"]              # (T, 3), float32
actions = episode["action"]               # (T, 3), float32
rewards = episode["reward"]               # (T,), float32
# Reward is the BotEvade task reward: -1 per capture, +1 on goal.
puffed = episode["puffed"]                # (T,), bool
capture_events = episode["capture_event"]
capture_count = episode["capture_count"]  # cumulative count at t
predator_sees_prey = episode["predator_sees_prey"]
goal_achieved = episode["goal_achieved"]
prey_predator_distance = episode["prey_predator_distance"]
goal_events = episode["goal_event"]
minimum_distance = episode["minimum_distance"]
# Canonical camera/geometry visibility labels.  Use pixels_visible as the
# visual risk-critic target; geometric_los is only a privileged diagnostic.
predator_geometric_los = episode["predator_geometric_los"]
predator_in_left_frustum = episode["predator_in_left_frustum"]
predator_in_right_frustum = episode["predator_in_right_frustum"]
predator_pixels_visible = episode["predator_pixels_visible"]
predator_within_detection_range = episode["predator_within_detection_range"]
predator_believed_visible = episode["predator_believed_visible"]
```

`predator_pixels_visible` is the camera-ground-truth label for a visual risk
critic. The legacy `predator_visible_camera` and `predator_visible_geometric`
arrays remain for compatibility, but should not be used in place of the
canonical fields.

`privileged_state` is saved only for trajectory diagnostics and evaluation; a vision-only VLA does not need to consume it. The Gym entry points and playable app automatically use `cellworld_cache/`, which is pre-populated for world `21_05`, so the documented first-person launches work offline.

The equivalent command-line launch is:

```bash
conda run -n Mice-BotEvade python -B mouse_play_app.py
```

---

---

### Training

The human-collection environment explicitly wires `reward.custom_reward`, so
the saved `reward` and episode `return` are not the default zero reward.
Reward callbacks receive the environment's named `reward_terms` mapping
(`capture`, `goal_achieved`, `goal_distance`, and related task terms), not the
flattened frame-stacked observation.

```bash
# SAC single-prey (BotEvade)
python SAC_train.py --config configs/sac_peeking_0406.yaml
```

### Evaluation

```bash
# Random-policy baseline on Oasis
python eval_oasis.py --episodes 20 --predator-ratio 0.15

# Render a few episodes
python eval_oasis.py --episodes 5 --render --predator-ratio 0.20

# Trained SAC checkpoint on BotEvade
python eval_sac.py --checkpoint runs/...
```

---


**`configs/`**

| File | Description |
|------|-------------|
| `sac_peeking_0406.yaml` | Single-prey SAC hyperparams. |
| `configs.py` | Config dataclasses. |
