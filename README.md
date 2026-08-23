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

### Quest / Slurm

For the verified Northwestern Quest setup, one-command submission workflow,
and job-management commands, see [QUEST_CONNECTION.md](QUEST_CONNECTION.md).

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

## PeekBench P0

PeekBench is the reproducible BotEvade benchmark for first-person active gaze,
threat memory, and exact-state safety branches.  Its engineering labels and
smoke results are not paper conclusions.  See
[the research audit](docs/RESEARCH_AUDIT.md) and
[the claims/test registry](docs/CLAIMS_AND_TESTS.md) for the information
boundary and falsifiable experiments.

Generate deterministic snapshots:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m benchmarks.peekbench generate \
  --config configs/peekbench/smoke.yaml
```

Run the no-key mock semantic evaluation:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m benchmarks.peekbench open-loop \
  --config configs/peekbench/smoke.yaml
```

Run fixed-gaze, candidate-gaze, and privileged-safe H-step branches:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m benchmarks.peekbench branches \
  --config configs/peekbench/smoke.yaml
```

Run EXP-00 Gaze Oracle Headroom with legal-duration head-yaw actions:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m benchmarks.peekbench headroom \
  --config configs/peekbench/exp00.yaml
```

EXP-00 does not train a model and never calls a remote provider. Fixed head,
random head, coverage scan, and privileged best gaze use the same deterministic
public-observation motion policy. The gaze oracle selects among complete
rollouts whose head movement was produced only by the legal normalized
`head_yaw_rate` action. The privileged safe controller is a separate
exact-state task upper reference. Results include per-run JSONL/CSV, method and
stratum tables, and a registered `GO`/`NO_GO` engineering screen in
`headroom_summary.json`.

Submit the registered 75-snapshot EXP-00 batch to Quest after committing and
pushing the current branch:

```bash
bash setup/submit_quest.sh setup/peekbench_exp00.sbatch
```

Run the EXP-01 perception--action gap engineering smoke without a provider
credential:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m benchmarks.peekbench exp01 \
  --config configs/peekbench/exp01_smoke.yaml
```

The smoke checks the paired measurement chain with the deterministic mock and
cannot support a VLM claim. The registered exploratory pilot uses
`configs/peekbench/exp01.yaml`, refuses mock fallback, and measures current
pixel detection, danger classification, legal reveal direction, 42 semantic
macro actions, current-only/public-history closed loops, paired capture, and
the conditional perception--action gap. See
[`docs/EXP01_PERCEPTION_ACTION_GAP_REPORT.md`](docs/EXP01_PERCEPTION_ACTION_GAP_REPORT.md)
for definitions, current status, and limitations.

After making `OPENROUTER_API_KEY` available only in the Quest job environment,
submit the pilot with:

```bash
bash setup/submit_quest.sh setup/peekbench_exp01.sbatch
```

Run the complete mock pipeline in one command:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m benchmarks.peekbench all \
  --config configs/peekbench/smoke.yaml
```

All artifacts are written under
`results/peekbench/<experiment_id>/`: resolved config, Git/environment
metadata, complete type-preserving state snapshots, public observations,
raw JSONL records, policy telemetry, and CSV summaries.

If `OPENROUTER_API_KEY` is absent, the deterministic local mock is selected
automatically.  To use OpenRouter, export the key only in the shell/session;
never write it to this repository, a config, a command log, or a test fixture.
The exact model and provider routing are versioned in the YAML config.  The
adapter uses strict structured output, bounded retry/timeout, and a local
response cache.

Audit human demonstrations without frame-level splitting:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m analysis.human_demo_audit \
  --data-root datasets/human_demos
```

With no recorded sessions, the audit writes a clear `no_data` summary and
exits successfully.

Run the EXP-02 human active-looking fingerprint analysis:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m analysis.human_active_looking_fingerprint \
  --data-root datasets/human_demos
```

The EXP-02 split manifest is grouped by `participant/session/world`; adjacent
frames from one video are never randomized across train/test partitions.  The
analysis writes descriptive strategy-structure metrics and an ethics reminder
under `results/human_active_looking_fingerprint/`.

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

The command above is the legacy task state-vector `MlpPolicy` baseline.
For a policy that actually consumes the recommended binocular public
observation and controls active head yaw, use the registered first-person SAC
pipeline. Because the observation is a Dict, Stable-Baselines3 uses
`MultiInputPolicy` with a shared-weight binocular CNN extractor rather than its
single-image `CnnPolicy` string:

```bash
# Tiny CPU gradient/checkpoint/load smoke.
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B train_first_person_sac.py \
  --config configs/sac_cnn_active_gaze.yaml \
  --smoke --output-root /tmp/mice-sac-smoke

# Quest: 1 A100, four environment workers, then paired evaluation and GIFs.
bash setup/submit_quest.sh setup/sac_cnn_train.sbatch
```

The training reward is explicitly registered in `reward.py`: goal event `+5`,
capture event `-5`, goal-distance shaping `-0.01*d`, and step cost `-0.001`.
Goal distance is available only to the environment reward callback; the policy
input remains exactly `image_left`, `image_right`, `proprio`, and
`previous_action`. Evaluation does not infer performance from shaped return: it
reports paired held-out clean-goal success, capture episodes, minimum distance,
path cost, and gaze use for the active policy, the same policy with head action
clamped to zero, and random actions. The job renders predeclared best,
median-ranked representative, and worst held-out episodes as split-screen
first-person/top-down GIFs.

Artifacts are written to
`results/sac/sac_cnn_active_gaze_<JOB_ID>/`, including resolved config and Git
metadata, TensorBoard logs, periodic/final/best checkpoints, evaluation JSONL
and CSV summaries, GIFs, and top-down trajectory PNGs.

Run the project-level 1,000-seed trajectory and density audit after a checkpoint
is frozen:

```bash
bash setup/submit_sac_cnn_density_1000.sh
```

Run EXP-05 on the same frozen checkpoint and held-out seeds to separate learned
active gaze from fixed camera placement at 0, +60, -60, and +30 degrees, plus a
fixed scan:

```bash
bash setup/submit_sac_gaze_ablation_1000.sh
```

The primary paired comparison is active gaze versus fixed +60 degrees. See
[`docs/EXP05_ACTIVE_GAZE_OR_CAMERA_POSE.md`](docs/EXP05_ACTIVE_GAZE_OR_CAMERA_POSE.md)
for the exact intervention and statistics.

The required follow-up is matched training across five training seeds per
condition on randomized, held-out task banks. It trains Active, Fixed 0,
Fixed +60, and preset scan from scratch; the old fixed start-goal route is not
used:

```bash
bash setup/submit_matched_training.sh
```

See
[`docs/MATCHED_TRAINING_GENERALIZATION.md`](docs/MATCHED_TRAINING_GENERALIZATION.md)
for the registered Train/Validation/Test region split, task manifest hashes,
training-seed statistics, and the remaining cross-layout/world blocker.

The default audit pairs active gaze, inference-time head clamping, and random
actions on seeds 1,000,000--1,000,999. It stores trajectories in flat,
pickle-free NPZ files and produces episode tables, paired bootstrap intervals,
trajectory overlays, episode-normalized occupancy density, capture density,
safety/efficiency distributions, head-yaw density, and a Markdown report. Each
episode contributes unit mass to the primary occupancy map, so slower methods
cannot dominate the density merely by taking more steps. Quest executes this as
30 independent single-process Slurm array shards (three methods by ten seed
shards), followed by an `afterok` aggregation job. This avoids sharing PyTorch
models through a local process pool and leaves each completed shard as an
auditable artifact.

The completed 1,000-seed results, spatial interpretation, and versioned compact
tables are documented in
[`docs/SAC_CNN_TRAJECTORY_DENSITY_REPORT.md`](docs/SAC_CNN_TRAJECTORY_DENSITY_REPORT.md).

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
