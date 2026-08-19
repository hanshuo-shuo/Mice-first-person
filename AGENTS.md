# Repository working guide

This repository is a first-person active-vision self-preservation research
platform built on Cellworld.  Changes must preserve the simulator semantics
while making experiments reproducible and falsifiable.

## Repository map

- `botevade_gym.py`, `oasis_gym.py`: Gymnasium task adapters.  Physics,
  reward events, termination, frame stacks, and task bookkeeping live here.
- `first_person.py`: binocular renderer and the first-person observation/action
  wrapper.  It adapts perception and egocentric controls; it does not own task
  physics or rewards.
- `cellworld_game-main/`: vendored simulator implementation.  Do not modify it
  for benchmark, policy, analysis, or documentation work.
- `mouse_play_app.py`: playable app and pickle-free human demonstration writer.
- `benchmarks/peekbench/`: deterministic snapshot and exact-state evaluation
  code.  PeekBench P0 targets `FirstPersonBotEvadeEnv`; Oasis does not yet have
  the complete snapshot API required for paired branches.
- `policies/`: vision-only policy interfaces and backends.  Policy code may
  receive only the public first-person observation and explicitly supplied
  public history.
- `analysis/`: offline dataset validation and descriptive analysis.
- `configs/peekbench/`: versioned experiment configurations.
- `results/peekbench/<experiment_id>/`: generated artifacts.  Do not treat
  generated smoke outputs as source data or paper evidence.
- `tests/`: contract, determinism, security-boundary, and smoke tests.

## Non-negotiable environment semantics

1. Preserve the recommended observation contract exactly:
   `image_left`, `image_right`, `proprio`, and `previous_action`.
2. Preserve the egocentric action contracts:
   `[forward_velocity, body_yaw_rate]` and, for active gaze,
   `[forward_velocity, body_yaw_rate, head_yaw_rate]`, all normalized to
   `[-1, 1]`.
3. Do not bypass point-mass integration, collision handling, predator
   navigation, reward callbacks, capture/goal events, or termination.
4. `AgentState.body_heading` is the canonical physical heading.  Head yaw is a
   separate wrapper state limited to +/-60 degrees by the current contract.
5. `predator_pixels_visible` is the canonical camera label.
   `predator_geometric_los`, predator coordinates, simulator state, and exact
   state dictionaries are privileged evaluation data.
6. Never place privileged data in a policy prompt, policy cache key material
   that is sent to a provider, or model-call logs.
7. Never read secrets from repository files.  OpenRouter credentials are read
   only from `OPENROUTER_API_KEY`; never print or serialize that value.
8. Snapshot branches must restore `get_state_dict()` before every branch and
   must not mutate the stored source state.
9. Do not claim a research hypothesis is verified from unit tests, generated
   snapshots, a smoke run, or preliminary engineering filters.

## Reproducible commands

Use the project Conda environment and disable bytecode writes because the
vendored simulator contains tracked historical bytecode files.

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy

conda run -n Mice-BotEvade pytest -q
python -m benchmarks.peekbench generate --config configs/peekbench/smoke.yaml
python -m benchmarks.peekbench open-loop --config configs/peekbench/smoke.yaml
python -m benchmarks.peekbench branches --config configs/peekbench/smoke.yaml
python -m analysis.human_demo_audit --data-root datasets/human_demos
```

The benchmark commands must work without `OPENROUTER_API_KEY`; that path uses
the deterministic mock vision policy.

## Required checks for changes

- Run the complete pytest suite, not only new tests.
- Run the no-key mock pipeline with a small config.
- Verify repeated generation with the same config/seed yields identical
  snapshot IDs.
- Verify state restore and paired branches are deterministic.
- Inspect generated policy-call JSONL for secret or privileged-field leakage.
- Keep generated benchmark outputs under `results/peekbench/`.
- Do not run large training jobs as part of development or acceptance.
